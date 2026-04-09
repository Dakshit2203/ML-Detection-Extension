"""
Tower B - Prototype 2 - backend/cache_sqlite.py

SQLite-backed cache for domain metadata results.

Why caching is necessary
Tower B probes three network signals per domain: DNS, TLS, and HTTP. On a fast connection, this typically takes
2–8 seconds. Without caching, every navigation to a previously-seen domain would incur this full cost, making the
extension unusably slow for normal browsing.

The cache stores the raw metadata dictionary (JSON-serialised) keyed by hostname. A TTL of 7 days is applied: results
older than 7 days are treated as expired and re-probed. This reflects the expectation that legitimate infrastructure
(DNS records, TLS certificates) changes slowly, whereas phishing domains are typically short-lived and would expire
from the cache naturally.

SQLite was chosen over an in-memory dict because this backend is restarted regularly - an in-memory cache would be
cleared on every restart. SQLite persists across restarts with no external dependencies.

Privacy note
The cache stores domain hostnames and their infrastructure metadata (IP count, TLS cert validity, HTTP headers). No
URLs, paths, or query strings are stored. No personal data is recorded.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheResult:
    """
    Wraps a cached metadata dictionary with its age in seconds.

    age_seconds is provided so that the backend can include cache freshness in the /predict debug response, which is useful during testing.
    """
    value: Dict[str, Any]
    age_seconds: float


class SQLiteCache:
    """
    Simple TTL-based key-value cache backed by a SQLite database.

    The table schema is intentionally minimal: a TEXT primary key, a JSON value column, and a Unix timestamp for TTL
    calculation. No migrations are needed - the table is created on first use.

    Parameters
    db_path : Path
    Path to the SQLite database file. Created automatically if absent.
    """

    CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            ts INTEGER NOT NULL
        )
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._initialise()

    def _initialise(self) -> None:
        """Creates the database file and table if they do not already exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute(self.CREATE_TABLE)
            con.commit()

    def get(self, key: str, ttl_seconds: int) -> Optional[CacheResult]:
        """
        Returns the cached value for key if it exists and has not expired.

        Parameters
        key: str The hostname to look up.
        ttl_seconds: int Maximum age (in seconds) before the result is stale.

        Returns
        CacheResult or None
            None if the key is absent or the cached result has expired.
        """
        now = int(time.time())
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT value, ts FROM cache WHERE key = ?", (key,)
            ).fetchone()

        if not row:
            return None

        value_str, ts = row
        age = now - int(ts)

        # Treat expired entries as a cache miss; the row is left in place and will be overwritten when the domain is re-probed.
        if age > ttl_seconds:
            return None

        try:
            return CacheResult(value=json.loads(value_str), age_seconds=float(age))
        except (json.JSONDecodeError, TypeError):
            # Corrupt cache entry - treat as a miss
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Inserts or replaces the cached value for key.

        INSERT OR REPLACE is used so that re-probing a domain always refreshes its cache entry and resets the TTL clock.

        Parameters
        ----------
        key: str The hostname being cached.
        value: Dict[str, Any] The metadata dictionary to store.
        """
        now = int(time.time())
        value_str = json.dumps(value, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
                (key, value_str, now)
            )
            con.commit()

    def delete(self, key: str) -> None:
        """Removes a specific entry from the cache. Used for testing."""
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM cache WHERE key = ?", (key,))
            con.commit()

    def clear(self) -> None:
        """Removes all cached entries. Used for testing."""
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM cache")
            con.commit()
