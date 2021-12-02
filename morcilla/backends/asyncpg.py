import json
import logging
import sys
import typing
import weakref

import asyncpg
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import hstore, pypostgresql
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.ddl import DDLElement

from morcilla.core import DatabaseURL
from morcilla.interfaces import ConnectionBackend, DatabaseBackend, TransactionBackend

logger = logging.getLogger("morcilla.backends.asyncpg")


class PostgresBackend(DatabaseBackend):
    # monkey-patch asyncpg to avoid introspection SQLs
    # please report your naming disgust to the authors of asyncpg
    _introspect_types_cache = (
        {}
    )  # type: typing.Dict[weakref.ReferenceType, typing.Dict[int, typing.Any]]
    _introspect_type_cache = (
        {}
    )  # type: typing.Dict[weakref.ReferenceType, typing.Dict[str, typing.Any]]
    _introspect_types_original = asyncpg.Connection._introspect_types
    _introspect_type_original = asyncpg.Connection._introspect_type

    def __init__(
        self, database_url: typing.Union[DatabaseURL, str], **options: typing.Any
    ) -> None:
        self._ignore_hstore = False
        self._introspect_types_cache_db = {}  # type: typing.Dict[int, typing.Any]
        self._introspect_type_cache_db = {}  # type: typing.Dict[str, typing.Any]

        self._database_url = DatabaseURL(database_url)
        self._options = {
            "init": self._register_codecs,  # enable HSTORE and JSON
            "statement_cache_size": 0,  # enable pgbouncer
        }
        self._options.update(options)
        self._dialect = self._get_dialect()
        self._pool = None

    def _get_dialect(self) -> Dialect:
        dialect = pypostgresql.dialect(paramstyle="pyformat")

        dialect.implicit_returning = True
        dialect.supports_native_enum = True
        dialect.supports_smallserial = True  # 9.2+
        dialect._backslash_escapes = False
        dialect.supports_sane_multi_rowcount = True  # psycopg 2.0.9+
        dialect._has_native_hstore = True
        dialect.supports_native_decimal = True

        return dialect

    def _get_connection_kwargs(self) -> dict:
        url_options = self._database_url.options

        kwargs = {}  # type: typing.Dict[str, typing.Any]
        min_size = url_options.get("min_size")
        max_size = url_options.get("max_size")
        ssl = url_options.get("ssl")

        if min_size is not None:
            kwargs["min_size"] = int(min_size)
        if max_size is not None:
            kwargs["max_size"] = int(max_size)
        if ssl is not None:
            kwargs["ssl"] = {"true": True, "false": False}[ssl.lower()]

        kwargs.update(self._options)

        return kwargs

    def _on_connection_close(self, conn: asyncpg.Connection) -> None:
        try:
            del self._introspect_types_cache[weakref.ref(conn)]
            del self._introspect_type_cache[weakref.ref(conn)]
        except TypeError:
            # conn is a PoolConnectionProxy and the DB is dying
            logger.warning(
                "could not dispose type introspection caches for connection %s", conn
            )

    async def _register_codecs(self, conn: asyncpg.Connection) -> None:
        # we have to maintain separate caches for each database because OID-s may be different
        self._introspect_types_cache[
            weakref.ref(conn)
        ] = self._introspect_types_cache_db
        self._introspect_type_cache[weakref.ref(conn)] = self._introspect_type_cache_db
        conn.add_termination_listener(self._on_connection_close)
        for flavor in ("json", "jsonb"):
            await conn.set_type_codec(
                flavor, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )
        if self._ignore_hstore:
            return
        try:
            await conn.set_builtin_type_codec("hstore", codec_name="pg_contrib.hstore")
        except ValueError:
            # no HSTORE is registered
            self._ignore_hstore = True
            logger.warning("no HSTORE is registered in %s", self._database_url)

    class _FakeStatement:
        name = ""

    @staticmethod
    async def _introspect_types_cached(
        self: asyncpg.Connection, typeoids: typing.Iterable[int], timeout: float
    ) -> typing.Any:
        introspect_types_cache = PostgresBackend._introspect_types_cache[
            weakref.ref(self)
        ]
        if missing := [oid for oid in typeoids if oid not in introspect_types_cache]:
            rows, stmt = await PostgresBackend._introspect_types_original(
                self, missing, timeout
            )
            assert stmt.name == ""
            for row in rows:
                introspect_types_cache[row["oid"]] = row
        return [
            introspect_types_cache[oid] for oid in typeoids
        ], PostgresBackend._FakeStatement

    @staticmethod
    async def _introspect_type_cached(
        self: asyncpg.Connection, typename: str, schema: str
    ) -> typing.Any:
        introspect_type_cache = PostgresBackend._introspect_type_cache[
            weakref.ref(self)
        ]
        try:
            return introspect_type_cache[typename]
        except KeyError:
            introspect_type_cache[
                typename
            ] = r = await PostgresBackend._introspect_type_original(
                self, typename, schema
            )
            return r

    async def connect(self) -> None:
        assert self._pool is None, "DatabaseBackend is already running"
        kwargs = dict(
            host=self._database_url.hostname,
            port=self._database_url.port,
            user=self._database_url.username,
            password=self._database_url.password,
            database=self._database_url.database,
        )
        kwargs.update(self._get_connection_kwargs())
        self._pool = await asyncpg.create_pool(**kwargs)

    async def disconnect(self) -> None:
        assert self._pool is not None, "DatabaseBackend is not running"
        await self._pool.close()
        self._pool = None

    def connection(self) -> "PostgresConnection":
        return PostgresConnection(self, self._dialect)


asyncpg.Connection._introspect_types = PostgresBackend._introspect_types_cached
asyncpg.Connection._introspect_type = PostgresBackend._introspect_type_cached

# monkey-patch HSTORE parser in SQLAlchemy
hstore = sys.modules[hstore.__module__]
_original_parse_hstore = hstore._parse_hstore


def _universal_parse_hstore(hstore_str: typing.Union[dict, str]) -> dict:
    if isinstance(hstore_str, dict):
        return hstore_str
    return _original_parse_hstore(hstore_str)


hstore._parse_hstore = _universal_parse_hstore


class PostgresConnection(ConnectionBackend):
    def __init__(self, database: PostgresBackend, dialect: Dialect):
        self._database = database
        self._dialect = dialect
        self._connection = None  # type: typing.Optional[asyncpg.Connection]

    async def acquire(self) -> None:
        assert self._connection is None, "Connection is already acquired"
        assert self._database._pool is not None, "DatabaseBackend is not running"
        self._connection = await self._database._pool.acquire()

    async def release(self) -> None:
        assert self._connection is not None, "Connection is not acquired"
        assert self._database._pool is not None, "DatabaseBackend is not running"
        self._connection = await self._database._pool.release(self._connection)
        self._connection = None

    async def fetch_all(self, query: ClauseElement) -> typing.List[typing.Sequence]:
        assert self._connection is not None, "Connection is not acquired"
        query_str, args = self._compile(query)
        return await self._connection.fetch(query_str, *args)

    async def fetch_one(self, query: ClauseElement) -> typing.Optional[typing.Sequence]:
        assert self._connection is not None, "Connection is not acquired"
        query_str, args = self._compile(query)
        return await self._connection.fetchrow(query_str, *args)

    async def fetch_val(self, query: ClauseElement, column: int = 0) -> typing.Any:
        assert self._connection is not None, "Connection is not acquired"
        query_str, args = self._compile(query)
        return await self._connection.fetchval(query_str, *args, column=column)

    async def execute(self, query: ClauseElement) -> typing.Any:
        return await self.fetch_val(query)

    async def execute_many_native(
        self, query: typing.Union[ClauseElement, str], values: list
    ) -> None:
        assert self._connection is not None, "Connection is not acquired"
        await self._connection.executemany(*self._compile(query, values))

    async def iterate(
        self, query: ClauseElement
    ) -> typing.AsyncGenerator[typing.Any, None]:
        assert self._connection is not None, "Connection is not acquired"
        query_str, args = self._compile(query)
        async for row in self._connection.cursor(query_str, *args):
            yield row

    def transaction(self) -> TransactionBackend:
        return PostgresTransaction(connection=self)

    def _compile(
        self,
        query: typing.Union[ClauseElement, str],
        values: typing.Optional[typing.List[typing.Mapping]] = None,
    ) -> typing.Tuple[str, typing.List[list]]:
        if isinstance(query, str):
            query = text(query)
        compiled = query.compile(
            dialect=self._dialect, compile_kwargs={"render_postcompile": True}
        )
        if not isinstance(query, DDLElement):
            if values:
                required_keys = values[0].keys()
            else:
                required_keys = compiled.params.keys()
            compiled_params = sorted((k, compiled.params[k]) for k in required_keys)
            sql_mapping = {
                key: "$" + str(i) for i, (key, _) in enumerate(compiled_params, start=1)
            }
            for key in compiled.params.keys() - required_keys:
                sql_mapping[key] = "DEFAULT"
            compiled_query = compiled.string % sql_mapping

            processors = compiled._bind_processors
            # we should not process HSTORE and JSON, asyncpg will do it for us
            removed = [
                key
                for key, val in processors.items()
                if val.__qualname__.startswith("HSTORE")
                or val.__qualname__.startswith("JSON")
            ]
            for key in removed:
                del processors[key]
            args = []
            if values is not None:
                param_mapping = {key: i for i, (key, _) in enumerate(compiled_params)}
                for dikt in values:
                    series = [None] * len(compiled_params)
                    args.append(series)
                    for key, val in dikt.items():
                        try:
                            val = processors[key](val)
                        except KeyError:
                            pass
                        series[param_mapping[key]] = val
            else:
                for key, val in compiled_params:
                    try:
                        val = processors[key](val)
                    except KeyError:
                        pass
                    args.append(val)
        else:
            compiled_query = compiled.string
            args = []
        return compiled_query, args

    @property
    def raw_connection(self) -> asyncpg.Connection:
        assert self._connection is not None, "Connection is not acquired"
        return self._connection


class PostgresTransaction(TransactionBackend):
    def __init__(self, connection: PostgresConnection):
        self._connection = connection
        self._transaction = (
            None
        )  # type: typing.Optional[asyncpg.transaction.Transaction]

    async def start(
        self, is_root: bool, extra_options: typing.Dict[typing.Any, typing.Any]
    ) -> None:
        assert self._connection._connection is not None, "Connection is not acquired"
        self._transaction = self._connection._connection.transaction(**extra_options)
        await self._transaction.start()

    async def commit(self) -> None:
        assert self._transaction is not None
        await self._transaction.commit()

    async def rollback(self) -> None:
        assert self._transaction is not None
        await self._transaction.rollback()
