"""
ChronoTrace / SynapseState Database Connection Manager
Provides robust SQLite connection lifecycle management, transaction support,
and dictionary-style row access with foreign keys strictly enforced.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# Default database location within the project repository
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "chronotrace.db"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

DB_PATH = Path(os.getenv("CHRONOTRACE_DB_PATH", str(DEFAULT_DB_PATH)))


def get_raw_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Creates and configures a raw SQLite connection.
    Enforces foreign key constraints and sets dictionary-style row factory.
    """
    target_path = Path(db_path) if db_path else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enforce SQLite foreign key validation
    conn.execute("PRAGMA foreign_keys = ON;")
    # Write-Ahead Logging for high concurrency in Streamlit
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def get_db(db_path: Optional[Union[str, Path]] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database transactions.
    Automatically commits on normal exit and rolls back on exception.
    """
    conn = get_raw_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise exc
    finally:
        conn.close()


def init_db(
    schema_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Executes DDL schema file to initialize or upgrade database tables.
    """
    schema_file = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found at: {schema_file}")

    with open(schema_file, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_db(db_path) as conn:
        conn.executescript(schema_sql)


def execute_query(
    sql: str,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Executes a SELECT query and returns rows as a list of standard dictionaries.
    """
    params = params or ()
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def execute_insert(
    sql: str,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    db_path: Optional[Union[str, Path]] = None
) -> int:
    """
    Executes an INSERT query and returns the last inserted row ID.
    """
    params = params or ()
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.lastrowid or 0


def execute_update(
    sql: str,
    params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    db_path: Optional[Union[str, Path]] = None
) -> int:
    """
    Executes an UPDATE/DELETE query and returns rowcount.
    """
    params = params or ()
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.rowcount


if __name__ == "__main__":
    print(f"Initializing database at: {DB_PATH}")
    init_db()
    print("Database schema successfully verified.")
