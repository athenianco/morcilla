import sqlalchemy as sa

from morcilla.backends.sqlite import SQLiteBackend


async def test_foreign_keys_always_enabled(tmp_path) -> None:
    """
    foreign_keys option is automatically enabled in the connection.
    """
    db_path = tmp_path / "db.sqlite3"
    url = f"sqlite:////{db_path}"
    backend = SQLiteBackend(url)
    conn = backend.connection()
    await conn.acquire()
    try:
        assert await conn.fetch_val(sa.text("PRAGMA foreign_keys")) == 1
    finally:
        await conn.release()
