"""
Unit tests for the backend connection arguments.
"""
import sys

import pytest

from morcilla.backends.asyncpg import PostgresBackend
from morcilla.core import DatabaseURL
from tests.test_databases import DATABASE_URLS, async_adapter


def test_postgres_pool_size():
    backend = PostgresBackend("postgres://localhost/database?min_size=1&max_size=20")
    kwargs = backend._get_connection_kwargs()
    for key, val in {"min_size": 1, "max_size": 20}.items():
        assert kwargs[key] == val


@async_adapter
async def test_postgres_pool_size_connect():
    for url in DATABASE_URLS:
        if DatabaseURL(url).dialect != "postgresql":
            continue
        backend = PostgresBackend(url + "?min_size=1&max_size=20")
        await backend.connect()
        await backend.disconnect()


def test_postgres_explicit_pool_size():
    backend = PostgresBackend("postgres://localhost/database", min_size=1, max_size=20)
    kwargs = backend._get_connection_kwargs()
    for key, val in {"min_size": 1, "max_size": 20}.items():
        assert kwargs[key] == val


def test_postgres_ssl():
    backend = PostgresBackend("postgres://localhost/database?ssl=true")
    kwargs = backend._get_connection_kwargs()
    assert kwargs["ssl"]


def test_postgres_explicit_ssl():
    backend = PostgresBackend("postgres://localhost/database", ssl=True)
    kwargs = backend._get_connection_kwargs()
    assert kwargs["ssl"]


def test_postgres_no_extra_options():
    backend = PostgresBackend("postgres://localhost/database")
    kwargs = backend._get_connection_kwargs()
    assert kwargs.keys() == {"init", "command_timeout"}


def test_postgres_password_as_callable():
    def gen_password():
        return "Foo"

    backend = PostgresBackend(
        "postgres://:password@localhost/database", password=gen_password
    )
    kwargs = backend._get_connection_kwargs()
    assert kwargs["password"] == gen_password
