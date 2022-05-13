import asyncio
import logging
import re
import typing
import uuid

import aiosqlite
from sqlalchemy.dialects.sqlite import pysqlite

try:
    from sqlalchemy.engine.cursor import CursorResultMetaData
    from sqlalchemy.engine.row import Row

    legacy_sqla = False
except ImportError:
    from sqlalchemy.engine.result import (
        ResultMetaData as CursorResultMetaData,
        RowProxy,
    )

    legacy_sqla = True
from sqlalchemy.engine.interfaces import Dialect, ExecutionContext
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.ddl import DDLElement

from morcilla.core import LOG_EXTRA, DatabaseURL
from morcilla.interfaces import ConnectionBackend, DatabaseBackend, TransactionBackend

logger = logging.getLogger("morcilla.backends.sqlite")


def _regexp(a: str, b: str) -> bool:
    return re.search(a, b) is not None


class SQLiteBackend(DatabaseBackend):
    def __init__(
        self, database_url: typing.Union[DatabaseURL, str], **options: typing.Any
    ) -> None:
        self._database_url = DatabaseURL(database_url)
        self._options = options
        self._dialect = pysqlite.dialect(paramstyle="qmark")
        # aiosqlite does not support decimals
        self._dialect.supports_native_decimal = False
        self._pool = SQLitePool(self._database_url, **self._options)
        self._transaction_lock = asyncio.Lock()

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        await self._pool.close()

    def connection(self) -> "SQLiteConnection":
        return SQLiteConnection(self._pool, self._dialect, self._transaction_lock)


class SQLitePool:
    def __init__(self, url: DatabaseURL, **options: typing.Any) -> None:
        self._url = url
        self._options = options
        self._global_connection = None  # type: typing.Optional[aiosqlite.Connection]

    async def acquire(self) -> aiosqlite.Connection:
        if (connection := self._global_connection) is None:
            connection = aiosqlite.connect(
                database=self._url.database, isolation_level=None, **self._options
            )
            await connection.__aenter__()
            await connection.create_function("regexp", 2, _regexp)
            await connection.execute("PRAGMA foreign_keys = true")
        if not self._url.database:
            self._global_connection = connection
        return connection

    async def release(self, connection: aiosqlite.Connection) -> None:
        if not self._url.database:
            return
        await connection.__aexit__(None, None, None)

    async def close(self) -> None:
        if not self._url.database and self._global_connection is not None:
            await self._global_connection.__aexit__(None, None, None)


class CompilationContext:
    def __init__(self, context: ExecutionContext):
        self.context = context


class SQLiteConnection(ConnectionBackend):
    def __init__(
        self, pool: SQLitePool, dialect: Dialect, transaction_lock: asyncio.Lock
    ):
        self._pool = pool
        self._dialect = dialect
        self._connection = None  # type: typing.Optional[aiosqlite.Connection]
        self._transaction_lock = transaction_lock

    async def acquire(self) -> None:
        assert self._connection is None, "Connection is already acquired"
        self._connection = await self._pool.acquire()

    async def release(self) -> None:
        assert self._connection is not None, "Connection is not acquired"
        await self._pool.release(self._connection)
        self._connection = None

    async def fetch_all(self, query: ClauseElement) -> typing.List[typing.Sequence]:
        assert self._connection is not None, "Connection is not acquired"
        query_str, args, context = self._compile(query)

        async with self._connection.execute(query_str, args) as cursor:
            rows = await cursor.fetchall()
            metadata = CursorResultMetaData(context, cursor.description)
            if not legacy_sqla:
                return [
                    Row(
                        metadata,
                        metadata._processors,
                        metadata._keymap,
                        Row._default_key_style,
                        row,
                    )
                    for row in rows
                ]
            return [
                RowProxy(
                    metadata,
                    row,
                    metadata._processors,
                    metadata._keymap,
                )
                for row in rows
            ]

    async def fetch_one(self, query: ClauseElement) -> typing.Optional[typing.Sequence]:
        assert self._connection is not None, "Connection is not acquired"
        query_str, args, context = self._compile(query)

        async with self._connection.execute(query_str, args) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            metadata = CursorResultMetaData(context, cursor.description)
            if not legacy_sqla:
                return Row(
                    metadata,
                    metadata._processors,
                    metadata._keymap,
                    Row._default_key_style,
                    row,
                )
            return RowProxy(
                metadata,
                row,
                metadata._processors,
                metadata._keymap,
            )

    async def execute(self, query: ClauseElement) -> typing.Any:
        assert self._connection is not None, "Connection is not acquired"
        if not self._connection.in_transaction:
            async with self._transaction_lock:
                return await self._execute(query)
        return await self._execute(query)

    async def _execute(self, query: ClauseElement) -> typing.Any:
        assert self._connection is not None, "Connection is not acquired"
        query_str, args, context = self._compile(query)
        async with self._connection.cursor() as cursor:
            await cursor.execute(query_str, args)
            if cursor.lastrowid == 0:
                return cursor.rowcount
            return cursor.lastrowid

    async def execute_many(self, queries: typing.List[ClauseElement]) -> None:
        assert self._connection is not None, "Connection is not acquired"
        for single_query in queries:
            await self.execute(single_query)

    async def iterate(
        self, query: ClauseElement
    ) -> typing.AsyncGenerator[typing.Any, None]:
        assert self._connection is not None, "Connection is not acquired"
        query_str, args, context = self._compile(query)
        async with self._connection.execute(query_str, args) as cursor:
            metadata = CursorResultMetaData(context, cursor.description)
            async for row in cursor:
                if not legacy_sqla:
                    yield Row(
                        metadata,
                        metadata._processors,
                        metadata._keymap,
                        Row._default_key_style,
                        row,
                    )
                else:
                    yield RowProxy(
                        metadata,
                        row,
                        metadata._processors,
                        metadata._keymap,
                    )

    def transaction(self) -> TransactionBackend:
        return SQLiteTransaction(self)

    def _compile(
        self, query: ClauseElement
    ) -> typing.Tuple[str, list, CompilationContext]:
        if legacy_sqla:
            compile_kwargs = {}
        else:
            compile_kwargs = {"render_postcompile": True}
        compiled = query.compile(dialect=self._dialect, compile_kwargs=compile_kwargs)

        execution_context = self._dialect.execution_ctx_cls()
        execution_context.dialect = self._dialect

        args = []

        if not isinstance(query, DDLElement):
            params = compiled.construct_params()
            for key in compiled.positiontup:
                raw_val = params[key]
                if key in compiled._bind_processors:
                    val = compiled._bind_processors[key](raw_val)
                else:
                    val = raw_val
                args.append(val)

            execution_context.result_column_struct = [
                compiled._result_columns,
                compiled._ordered_columns,
                compiled._textual_ordered_columns,
            ]
            if not legacy_sqla:
                execution_context.result_column_struct.append(
                    compiled._loose_column_name_matching
                )

        query_message = compiled.string.replace(" \n", " ").replace("\n", " ")
        logger.debug(
            "Query: %s Args: %s", query_message, repr(tuple(args)), extra=LOG_EXTRA
        )
        return compiled.string, args, CompilationContext(execution_context)

    @property
    def raw_connection(self) -> aiosqlite.core.Connection:
        assert self._connection is not None, "Connection is not acquired"
        return self._connection


class SQLiteTransaction(TransactionBackend):
    def __init__(self, connection: SQLiteConnection):
        self._connection = connection
        self._is_root = False
        self._locked = False
        self._savepoint_name = ""

    async def start(
        self, is_root: bool, extra_options: typing.Dict[typing.Any, typing.Any]
    ) -> None:
        assert self._connection._connection is not None, "Connection is not acquired"
        self._is_root = is_root
        if self._is_root:
            assert not self._locked
            await self._connection._transaction_lock.acquire()
            self._locked = True
            async with self._connection._connection.execute("BEGIN") as cursor:
                await cursor.close()
        else:
            id = str(uuid.uuid4()).replace("-", "_")
            self._savepoint_name = f"STARLETTE_SAVEPOINT_{id}"
            async with self._connection._connection.execute(
                f"SAVEPOINT {self._savepoint_name}"
            ) as cursor:
                await cursor.close()

    async def commit(self) -> None:
        assert self._connection._connection is not None, "Connection is not acquired"
        if self._is_root:
            assert self._locked
            async with self._connection._connection.execute("COMMIT") as cursor:
                await cursor.close()
            self._locked = False
            self._connection._transaction_lock.release()
        else:
            async with self._connection._connection.execute(
                f"RELEASE SAVEPOINT {self._savepoint_name}"
            ) as cursor:
                await cursor.close()

    async def rollback(self) -> None:
        assert self._connection._connection is not None, "Connection is not acquired"
        if self._is_root:
            assert self._locked
            async with self._connection._connection.execute("ROLLBACK") as cursor:
                await cursor.close()
            self._locked = False
            self._connection._transaction_lock.release()
        else:
            async with self._connection._connection.execute(
                f"ROLLBACK TO SAVEPOINT {self._savepoint_name}"
            ) as cursor:
                await cursor.close()
