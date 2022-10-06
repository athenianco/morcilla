import asyncio
import contextlib
import functools
import logging
import typing
from types import TracebackType
from urllib.parse import SplitResult, parse_qsl, unquote, urlsplit

from sqlalchemy import text
from sqlalchemy.sql import ClauseElement

from morcilla.importer import import_from_string
from morcilla.interfaces import ConnectionBackend, DatabaseBackend, TransactionBackend

try:  # pragma: no cover
    import click

    # Extra log info for optional coloured terminal outputs.
    LOG_EXTRA = {
        "color_message": "Query: " + click.style("%s", bold=True) + " Args: %s"
    }
    CONNECT_EXTRA = {
        "color_message": "Connected to database " + click.style("%s", bold=True)
    }
    DISCONNECT_EXTRA = {
        "color_message": "Disconnected from database " + click.style("%s", bold=True)
    }
except ImportError:  # pragma: no cover
    LOG_EXTRA = {}
    CONNECT_EXTRA = {}
    DISCONNECT_EXTRA = {}


logger = logging.getLogger("morcilla")


class Database:
    SUPPORTED_BACKENDS = {
        "postgresql": "morcilla.backends.asyncpg:PostgresBackend",
        "postgresql+aiopg": "morcilla.backends.aiopg:AiopgBackend",
        "mysql": "morcilla.backends.mysql:MySQLBackend",
        "mysql+asyncmy": "morcilla.backends.asyncmy:AsyncMyBackend",
        "sqlite": "morcilla.backends.sqlite:SQLiteBackend",
    }

    def __init__(
        self,
        url: typing.Union[str, "DatabaseURL"],
        **options: typing.Any,
    ):
        self.url = DatabaseURL(url)
        self.options = options
        self._is_connected = False

        backend_str = self._get_backend()
        backend_cls = import_from_string(backend_str)
        assert issubclass(backend_cls, DatabaseBackend)
        self._backend = backend_cls(self.url, **self.options)

    def __repr__(self) -> str:
        """Support repr()."""
        return f"Database('{self.url.obscure_password}', **{self.options})"

    def __str__(self) -> str:
        """Support str()."""
        return repr(self)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def connect(self) -> "Database":
        """
        Establish the connection pool.
        """
        if self._is_connected:
            logger.debug("Already connected, skipping connection")
            return self

        await self._backend.connect()
        logger.info(
            "Connected to database %s", self.url.obscure_password, extra=CONNECT_EXTRA
        )
        self._is_connected = True
        return self

    async def disconnect(self) -> None:
        """
        Close all connections in the connection pool.
        """
        if not self._is_connected:
            logger.debug("Already disconnected, skipping disconnection")
            return None

        await self._backend.disconnect()
        logger.info(
            "Disconnected from database %s",
            self.url.obscure_password,
            extra=DISCONNECT_EXTRA,
        )
        self._is_connected = False

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: typing.Type[BaseException] = None,
        exc_value: BaseException = None,
        traceback: TracebackType = None,
    ) -> None:
        await self.disconnect()

    async def fetch_all(
        self, query: typing.Union[ClauseElement, str], values: dict = None
    ) -> typing.List[typing.Sequence]:
        async with self.connection() as connection:
            return await connection.fetch_all(query, values)

    async def fetch_one(
        self, query: typing.Union[ClauseElement, str], values: dict = None
    ) -> typing.Optional[typing.Sequence]:
        async with self.connection() as connection:
            return await connection.fetch_one(query, values)

    async def fetch_val(
        self,
        query: typing.Union[ClauseElement, str],
        values: dict = None,
        column: typing.Any = 0,
    ) -> typing.Any:
        async with self.connection() as connection:
            return await connection.fetch_val(query, values, column=column)

    async def execute(
        self, query: typing.Union[ClauseElement, str], values: dict = None
    ) -> typing.Any:
        async with self.connection() as connection:
            return await connection.execute(query, values)

    async def execute_many(
        self, query: typing.Union[ClauseElement, str], values: list
    ) -> None:
        async with self.connection() as connection:
            return await connection.execute_many(query, values)

    async def iterate(
        self, query: typing.Union[ClauseElement, str], values: dict = None
    ) -> typing.AsyncGenerator[typing.Mapping, None]:
        async with self.connection() as connection:
            async for record in connection.iterate(query, values):
                yield record

    def connection(self) -> "Connection":
        return Connection(self._backend)

    def _get_backend(self) -> str:
        return self.SUPPORTED_BACKENDS.get(
            self.url.scheme, self.SUPPORTED_BACKENDS[self.url.dialect]
        )


class Connection:
    def __init__(self, backend: DatabaseBackend) -> None:
        self._backend = backend

        self._connection_lock = asyncio.Lock()
        self._connection = self._backend.connection()  # type: ConnectionBackend
        self._connection_counter = 0

        self._transaction_lock = asyncio.Lock()
        self._transaction_stack = []  # type: typing.List[Transaction]

        self._query_lock = asyncio.Lock()

    async def __aenter__(self) -> "Connection":
        async with self._connection_lock:
            self._connection_counter += 1
            try:
                if self._connection_counter == 1:
                    await self._connection.acquire()
            except Exception as e:
                self._connection_counter -= 1
                raise e
        return self

    async def __aexit__(
        self,
        exc_type: typing.Type[BaseException] = None,
        exc_value: BaseException = None,
        traceback: TracebackType = None,
    ) -> None:
        async with self._connection_lock:
            assert self._connection is not None
            self._connection_counter -= 1
            if self._connection_counter == 0:
                await self._connection.release()

    async def fetch_all(
        self, query: typing.Union[ClauseElement, str], values: dict = None
    ) -> typing.List[typing.Sequence]:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_all(built_query)

    async def fetch_one(
        self, query: typing.Union[ClauseElement, str], values: dict = None
    ) -> typing.Optional[typing.Sequence]:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_one(built_query)

    async def fetch_val(
        self,
        query: typing.Union[ClauseElement, str],
        values: dict = None,
        column: typing.Any = 0,
    ) -> typing.Any:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.fetch_val(built_query, column)

    async def execute(
        self, query: typing.Union[ClauseElement, str], values: dict = None
    ) -> typing.Any:
        built_query = self._build_query(query, values)
        async with self._query_lock:
            return await self._connection.execute(built_query)

    async def execute_many(
        self, query: typing.Union[ClauseElement, str], values: list
    ) -> None:
        if not values:
            return
        try:
            return await self._connection.execute_many_native(query, values)
        except NotImplementedError:
            queries = [self._build_query(query, values_set) for values_set in values]
            async with self._query_lock:
                return await self._connection.execute_many(queries)

    async def iterate(
        self, query: typing.Union[ClauseElement, str], values: dict = None
    ) -> typing.AsyncGenerator[typing.Any, None]:
        built_query = self._build_query(query, values)
        async with self.transaction():
            async with self._query_lock:
                async for record in self._connection.iterate(built_query):
                    yield record

    def transaction(
        self, *, force_rollback: bool = False, **kwargs: typing.Any
    ) -> "Transaction":
        return Transaction(self, force_rollback, **kwargs)

    @contextlib.asynccontextmanager
    async def raw_connection(self) -> typing.AsyncIterator:
        """
        Return the underlying DB backend's connection.

        For example, it is `asyncpg.Connection` for asyncpg backend.

        We acquire the connection lock while the raw connection is being used. Sample code:
        ```
        async with connection.raw_connection() as raw_conn:
            print(await raw_conn.fetchval("SELECT true"))
        ```
        """
        async with self._query_lock:
            yield self._connection.raw_connection

    @staticmethod
    def _build_query(
        query: typing.Union[ClauseElement, str], values: dict = None
    ) -> ClauseElement:
        if isinstance(query, str):
            query = text(query)

            return query.bindparams(**values) if values is not None else query
        elif values:
            return query.values(**values)

        return query


class Transaction:
    def __init__(
        self,
        connection: Connection,
        force_rollback: bool,
        **kwargs: typing.Any,
    ) -> None:
        self._connection = connection
        self._force_rollback = force_rollback
        self._extra_options = kwargs

    async def __aenter__(self) -> "Transaction":
        """
        Called when entering `async with database.transaction()`
        """
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: typing.Type[BaseException] = None,
        exc_value: BaseException = None,
        traceback: TracebackType = None,
    ) -> None:
        """
        Called when exiting `async with database.transaction()`
        """
        if exc_type is not None or self._force_rollback:
            await self.rollback()
        else:
            await self.commit()

    def __await__(self) -> typing.Generator:
        """
        Called if using the low-level `transaction = await database.transaction()`
        """
        return self.start().__await__()

    def __call__(self, func: typing.Callable) -> typing.Callable:
        """
        Called if using `@database.transaction()` as a decorator.
        """

        @functools.wraps(func)
        async def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            async with self:
                return await func(*args, **kwargs)

        return wrapper

    async def start(self) -> "Transaction":
        self._transaction = (
            self._connection._connection.transaction()
        )  # type: TransactionBackend

        async with self._connection._transaction_lock:
            is_root = not self._connection._transaction_stack
            await self._connection.__aenter__()
            try:
                await self._transaction.start(
                    is_root=is_root, extra_options=self._extra_options
                )
            except Exception as e:
                await self._connection.__aexit__()
                raise e from None
            self._connection._transaction_stack.append(self)
        return self

    async def commit(self) -> None:
        async with self._connection._transaction_lock:
            assert self._connection._transaction_stack[-1] is self
            self._connection._transaction_stack.pop()
            await self._transaction.commit()
            await self._connection.__aexit__()

    async def rollback(self) -> None:
        async with self._connection._transaction_lock:
            assert self._connection._transaction_stack[-1] is self
            self._connection._transaction_stack.pop()
            await self._transaction.rollback()
            await self._connection.__aexit__()


class _EmptyNetloc(str):
    def __bool__(self) -> bool:
        return True


class DatabaseURL:
    def __init__(self, url: typing.Union[str, "DatabaseURL"]):
        if isinstance(url, DatabaseURL):
            self._url: str = url._url
        elif isinstance(url, str):
            self._url = url
        else:
            raise TypeError(
                f"Invalid type for DatabaseURL. Expected str or DatabaseURL, got {type(url)}"
            )

    @property
    def components(self) -> SplitResult:
        if not hasattr(self, "_components"):
            self._components = urlsplit(self._url)
        return self._components

    @property
    def scheme(self) -> str:
        return self.components.scheme

    @property
    def dialect(self) -> str:
        return self.components.scheme.split("+")[0]

    @property
    def driver(self) -> str:
        if "+" not in self.components.scheme:
            return ""
        return self.components.scheme.split("+", 1)[1]

    @property
    def userinfo(self) -> typing.Optional[bytes]:
        if self.components.username:
            info = self.components.username
            if self.components.password:
                info += ":" + self.components.password
            return info.encode("utf-8")
        return None

    @property
    def username(self) -> typing.Optional[str]:
        if self.components.username is None:
            return None
        return unquote(self.components.username)

    @property
    def password(self) -> typing.Optional[str]:
        if self.components.password is None:
            return None
        return unquote(self.components.password)

    @property
    def hostname(self) -> typing.Optional[str]:
        return (
            self.components.hostname
            or self.options.get("host")
            or self.options.get("unix_sock")
        )

    @property
    def port(self) -> typing.Optional[int]:
        return self.components.port

    @property
    def netloc(self) -> typing.Optional[str]:
        return self.components.netloc

    @property
    def database(self) -> str:
        path = self.components.path
        if path.startswith("/"):
            path = path[1:]
        return unquote(path)

    @property
    def options(self) -> dict:
        if not hasattr(self, "_options"):
            self._options = dict(parse_qsl(self.components.query))
        return self._options

    def replace(self, **kwargs: typing.Any) -> "DatabaseURL":
        if (
            "username" in kwargs
            or "password" in kwargs
            or "hostname" in kwargs
            or "port" in kwargs
        ):
            hostname = kwargs.pop("hostname", self.hostname)
            port = kwargs.pop("port", self.port)
            username = kwargs.pop("username", self.components.username)
            password = kwargs.pop("password", self.components.password)

            netloc = hostname
            if port is not None:
                netloc += f":{port}"
            if username is not None:
                userpass = username
                if password is not None:
                    userpass += f":{password}"
                netloc = f"{userpass}@{netloc}"

            kwargs["netloc"] = netloc

        if "database" in kwargs:
            kwargs["path"] = "/" + kwargs.pop("database")

        if "dialect" in kwargs or "driver" in kwargs:
            dialect = kwargs.pop("dialect", self.dialect)
            driver = kwargs.pop("driver", self.driver)
            kwargs["scheme"] = f"{dialect}+{driver}" if driver else dialect

        if not kwargs.get("netloc", self.netloc):
            # Using an empty string that evaluates as True means we end up
            # with URLs like `sqlite:///database` instead of `sqlite:/database`
            kwargs["netloc"] = _EmptyNetloc()

        components = self.components._replace(**kwargs)
        return self.__class__(components.geturl())

    @property
    def obscure_password(self) -> "DatabaseURL":
        if self.password:
            return self.replace(password="<secured>")
        return self

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.obscure_password._url)})"

    def __eq__(self, other: typing.Any) -> bool:
        return str(self) == str(other)
