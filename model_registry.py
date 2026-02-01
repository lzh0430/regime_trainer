"""
Model version registry - SQLite-backed version dirs and PROD pointer per (symbol, timeframe).
"""
import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Default DB path (caller can override via init or env)
def _default_db_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "model_registry.db")

def _default_models_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models")


def _get_conn(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def _init_database(db_path: str) -> None:
    with _get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prod_pointer (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                version_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (symbol, timeframe)
            );
            CREATE TABLE IF NOT EXISTS training_configs (
                config_version_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                created_by TEXT,
                description TEXT,
                is_active INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS training_config_data (
                config_version_id TEXT NOT NULL,
                config_key TEXT NOT NULL,
                config_value TEXT NOT NULL,
                PRIMARY KEY (config_version_id, config_key),
                FOREIGN KEY (config_version_id) REFERENCES training_configs(config_version_id)
            );
            CREATE TABLE IF NOT EXISTS model_config_links (
                model_version_id TEXT NOT NULL,
                config_version_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (model_version_id, symbol, timeframe),
                FOREIGN KEY (config_version_id) REFERENCES training_configs(config_version_id)
            );
        """)
        conn.commit()


def allocate_version_id(models_dir: Optional[str] = None, db_path: Optional[str] = None) -> str:
    """
    Allocate a new version id: yyyy-MM-dd-index where index is per-day auto-increment.

    Args:
        models_dir: Base models directory (default: project models/).
        db_path: Registry DB path (default: data/model_registry.db).

    Returns:
        Version id string, e.g. "2025-01-31-1".
    """
    models_dir = models_dir or _default_models_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    prefix = f"{today}-"
    indexes = []
    if os.path.isdir(models_dir):
        for name in os.listdir(models_dir):
            if name.startswith(prefix) and os.path.isdir(os.path.join(models_dir, name)):
                try:
                    idx_part = name[len(prefix):]
                    if idx_part.isdigit():
                        indexes.append(int(idx_part))
                except ValueError:
                    continue
    next_index = max(indexes, default=0) + 1
    version_id = f"{today}-{next_index}"
    return version_id


def register_version(version_id: str, db_path: Optional[str] = None) -> None:
    """
    Register a version in model_versions (idempotent).

    Args:
        version_id: e.g. "2025-01-31-1".
        db_path: Registry DB path.
    """
    db_path = db_path or _default_db_path()
    _init_database(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO model_versions (version_id, created_at) VALUES (?, ?)",
            (version_id, now),
        )
        conn.commit()
    logger.debug(f"Registered version: {version_id}")


def set_prod(
    symbol: str,
    timeframe: str,
    version_id: str,
    models_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """
    Set PROD pointer for (symbol, timeframe) to version_id.
    Validates that models/{version_id}/{symbol}/{timeframe}/ exists.

    Returns:
        True if set, False if validation failed (path does not exist).
    """
    models_dir = models_dir or _default_models_dir()
    db_path = db_path or _default_db_path()
    dir_path = os.path.join(models_dir, version_id, symbol, timeframe)
    if not os.path.isdir(dir_path):
        logger.warning(f"PROD set failed: path does not exist: {dir_path}")
        return False
    _init_database(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO prod_pointer (symbol, timeframe, version_id, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe) DO UPDATE SET
                version_id = excluded.version_id,
                updated_at = excluded.updated_at
            """,
            (symbol, timeframe, version_id, now),
        )
        conn.commit()
    logger.info(f"PROD set: {symbol} {timeframe} -> {version_id}")
    return True


def get_prod_info(
    symbol: str,
    timeframe: str,
    models_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get PROD pointer row for (symbol, timeframe): symbol, timeframe, version_id, updated_at.
    Returns None if no row in prod_pointer (caller can use get_prod_version for effective version).
    """
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return None
    _init_database(db_path)
    with _get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT symbol, timeframe, version_id, updated_at FROM prod_pointer WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {"symbol": row["symbol"], "timeframe": row["timeframe"], "version_id": row["version_id"], "updated_at": row["updated_at"]}


def get_prod_version(
    symbol: str,
    timeframe: str,
    models_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[str]:
    """
    Get PROD version_id for (symbol, timeframe).
    If no row in prod_pointer, returns latest version that has that (symbol, timeframe) on disk.
    If none, returns None (caller can fall back to legacy path).
    """
    db_path = db_path or _default_db_path()
    models_dir = models_dir or _default_models_dir()
    if not os.path.isfile(db_path):
        return get_latest_version(symbol, timeframe, models_dir, db_path)
    _init_database(db_path)
    with _get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT version_id FROM prod_pointer WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        row = cur.fetchone()
    if row is not None:
        return row["version_id"]
    return get_latest_version(symbol, timeframe, models_dir, db_path)


def get_latest_version(
    symbol: str,
    timeframe: str,
    models_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[str]:
    """
    Find latest version_id that has models/{version_id}/{symbol}/{timeframe}/ on disk.
    Prefer ordering by model_versions.created_at if DB exists; else by version_id string sort.
    """
    models_dir = models_dir or _default_models_dir()
    db_path = db_path or _default_db_path()
    if not os.path.isdir(models_dir):
        return None
    # Collect version dirs that contain symbol/timeframe
    candidates = []
    for vname in os.listdir(models_dir):
        vpath = os.path.join(models_dir, vname)
        if not os.path.isdir(vpath):
            continue
        target = os.path.join(vpath, symbol, timeframe)
        if os.path.isdir(target):
            candidates.append(vname)
    if not candidates:
        return None
    # Prefer DB created_at order if we have it
    if os.path.isfile(db_path):
        _init_database(db_path)
        with _get_conn(db_path) as conn:
            placeholders = ",".join("?" * len(candidates))
            cur = conn.execute(
                f"SELECT version_id, created_at FROM model_versions WHERE version_id IN ({placeholders}) ORDER BY created_at DESC",
                candidates,
            )
            ordered = [row[0] for row in cur.fetchall()]
        # Any candidate not in DB, append at end
        for c in candidates:
            if c not in ordered:
                ordered.append(c)
        return ordered[0] if ordered else None
    # No DB: sort by version_id (yyyy-MM-dd-N gives chronological order)
    candidates.sort(reverse=True)
    return candidates[0]


def list_versions(
    models_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all version_ids and which (symbol, timeframe) each contains.
    Optionally include which (symbol, timeframe) are PROD and their version.
    Also includes config_version_id for each model if available.
    """
    models_dir = models_dir or _default_models_dir()
    db_path = db_path or _default_db_path()
    result = []
    if not os.path.isdir(models_dir):
        return result
    prod_map = {}
    if os.path.isfile(db_path):
        _init_database(db_path)
        with _get_conn(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT symbol, timeframe, version_id, updated_at FROM prod_pointer"):
                prod_map[(row["symbol"], row["timeframe"])] = {
                    "version_id": row["version_id"],
                    "updated_at": row["updated_at"],
                }
    version_created = {}
    if os.path.isfile(db_path):
        with _get_conn(db_path) as conn:
            for row in conn.execute("SELECT version_id, created_at FROM model_versions"):
                version_created[row[0]] = row[1]
    
    # Load config version mappings for all models
    config_map = {}  # {(model_version_id, symbol, timeframe): config_version_id}
    if os.path.isfile(db_path):
        with _get_conn(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT model_version_id, symbol, timeframe, config_version_id FROM model_config_links"):
                config_map[(row["model_version_id"], row["symbol"], row["timeframe"])] = row["config_version_id"]
    
    for vname in sorted(os.listdir(models_dir), reverse=True):
        vpath = os.path.join(models_dir, vname)
        if not os.path.isdir(vpath):
            continue
        symbols = []
        for sym in os.listdir(vpath):
            sym_path = os.path.join(vpath, sym)
            if not os.path.isdir(sym_path):
                continue
            for tf in os.listdir(sym_path):
                tf_path = os.path.join(sym_path, tf)
                if os.path.isdir(tf_path):
                    is_prod = prod_map.get((sym, tf), {}).get("version_id") == vname
                    config_version_id = config_map.get((vname, sym, tf))
                    model_info = {
                        "symbol": sym,
                        "timeframe": tf,
                        "is_prod": is_prod
                    }
                    if config_version_id:
                        model_info["config_version_id"] = config_version_id
                    symbols.append(model_info)
        result.append({
            "version_id": vname,
            "created_at": version_created.get(vname),
            "symbols": symbols,  # Keep for backward compatibility
            "contents": symbols,  # Frontend expects 'contents'
        })
    return result
