"""
Forward testing: post-training hook, timeframe-based runs, golden source comparison, evidence tables.
"""
import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Protocol

logger = logging.getLogger(__name__)


def _default_db_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "model_registry.db")


def _get_conn(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def _init_forward_test_tables(db_path: str) -> None:
    with _get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS forward_test_campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                created_at TEXT NOT NULL,
                required_runs INTEGER NOT NULL DEFAULT 5,
                status TEXT NOT NULL DEFAULT 'active',
                UNIQUE(version_id, symbol, timeframe)
            );
            CREATE TABLE IF NOT EXISTS forward_test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                run_at TEXT NOT NULL,
                model_regime TEXT NOT NULL,
                golden_regime TEXT,
                match INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (campaign_id) REFERENCES forward_test_campaigns(id)
            );
        """)
        conn.commit()


# --------------- Golden source ---------------

class GoldenSource(Protocol):
    """Protocol for golden truth: get_golden_regime(symbol, timeframe, timestamp) -> regime name or None."""

    def get_golden_regime(self, symbol: str, timeframe: str, timestamp: datetime) -> Optional[str]:
        ...


class DefaultGoldenSource:
    """Default: no golden source; always returns None."""

    def get_golden_regime(self, symbol: str, timeframe: str, timestamp: datetime) -> Optional[str]:
        return None


# --------------- Campaign enrollment ---------------

def enroll_campaign(
    version_id: str,
    symbol: str,
    timeframe: str,
    required_runs: int = 5,
    db_path: Optional[str] = None,
) -> None:
    """
    Enroll (version_id, symbol, timeframe) for forward testing. Idempotent.
    """
    db_path = db_path or _default_db_path()
    _init_forward_test_tables(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO forward_test_campaigns
            (version_id, symbol, timeframe, created_at, required_runs, status)
            VALUES (?, ?, ?, ?, ?, 'active')
            """,
            (version_id, symbol, timeframe, now, required_runs),
        )
        conn.commit()
    logger.info(f"Forward test campaign enrolled: {version_id} {symbol} {timeframe}")


def on_training_finished(
    symbol: str,
    primary_timeframe: str,
    version_id: str,
    config: Any,
    db_path: Optional[str] = None,
) -> None:
    """
    Hook to call at end of full_retrain / incremental_train. Enrolls campaign with required_runs from config.
    """
    required_runs = getattr(config, "FORWARD_TEST_REQUIRED_RUNS", 5)
    if db_path is None:
        base = getattr(config, "DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
        db_path = os.path.join(base, "model_registry.db")
    enroll_campaign(version_id, symbol, primary_timeframe, required_runs=required_runs, db_path=db_path)


# --------------- Run one forward test ---------------

def _campaign_run_count(conn: sqlite3.Connection, campaign_id: int) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM forward_test_runs WHERE campaign_id = ?", (campaign_id,))
    return cur.fetchone()[0]


def run_one_forward_test(
    version_id: str,
    symbol: str,
    timeframe: str,
    golden_source: Optional[GoldenSource] = None,
    db_path: Optional[str] = None,
    config: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Run one forward test: predict with version_id, optionally compare with golden, record run.
    Returns run record dict or None if campaign not found / not active / already has required_runs.
    """
    db_path = db_path or _default_db_path()
    _init_forward_test_tables(db_path)
    golden_source = golden_source or DefaultGoldenSource()

    with _get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, required_runs, status FROM forward_test_campaigns WHERE version_id = ? AND symbol = ? AND timeframe = ?",
            (version_id, symbol, timeframe),
        )
        row = cur.fetchone()
    if row is None:
        logger.debug(f"No forward test campaign for {version_id} {symbol} {timeframe}")
        return None
    campaign_id = row["id"]
    required_runs = row["required_runs"]
    status = row["status"]
    if status != "active":
        logger.debug(f"Campaign {campaign_id} not active: {status}")
        return None

    with _get_conn(db_path) as conn:
        n = _campaign_run_count(conn, campaign_id)
    if n >= required_runs:
        logger.debug(f"Campaign {campaign_id} already has {n} runs")
        return None

    # Predict with version
    if config is None:
        from config import TrainingConfig
        config = TrainingConfig
    from model_api import ModelAPI
    api = ModelAPI(config)
    try:
        result = api.predict_next_regime_for_version(symbol, timeframe, version_id)
    except Exception as e:
        logger.error(f"Forward test predict failed: {e}", exc_info=True)
        return None

    model_regime = result["most_likely_regime"]["name"]
    run_at = datetime.utcnow().isoformat() + "Z"
    golden_regime = golden_source.get_golden_regime(symbol, timeframe, datetime.utcnow())
    match_val = None
    if golden_regime is not None:
        match_val = 1 if (model_regime.strip() == golden_regime.strip()) else 0

    with _get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (campaign_id, run_at, model_regime, golden_regime, match_val, run_at),
        )
        conn.commit()
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        new_count = _campaign_run_count(conn, campaign_id)
        if new_count >= required_runs:
            conn.execute("UPDATE forward_test_campaigns SET status = 'qualified' WHERE id = ?", (campaign_id,))
            conn.commit()
            logger.info(f"Forward test campaign {campaign_id} qualified ({new_count} runs)")

    return {
        "campaign_id": campaign_id,
        "run_at": run_at,
        "model_regime": model_regime,
        "golden_regime": golden_regime,
        "match": match_val,
        "runs_count": new_count,
        "required_runs": required_runs,
    }


# --------------- List active campaigns for a timeframe ---------------

def get_active_campaigns_for_timeframe(
    timeframe: str,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return list of active campaigns with run_count < required_runs for the given timeframe."""
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return []
    _init_forward_test_tables(db_path)
    with _get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, version_id, symbol, timeframe, required_runs FROM forward_test_campaigns WHERE timeframe = ? AND status = 'active'",
            (timeframe,),
        )
        rows = cur.fetchall()
    out = []
    for row in rows:
        with _get_conn(db_path) as conn2:
            n = _campaign_run_count(conn2, row["id"])
        if n < row["required_runs"]:
            out.append({
                "id": row["id"],
                "version_id": row["version_id"],
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "required_runs": row["required_runs"],
                "run_count": n,
            })
    return out


# --------------- Scheduler ---------------

class ForwardTestScheduler:
    """
    Runs forward tests at timeframe-aligned intervals (5m -> every 5 min, 15m -> every 15 min, 1h -> every 1h).
    """

    def __init__(self, config: Any = None, golden_source: Optional[GoldenSource] = None):
        if config is None:
            from config import TrainingConfig
            config = TrainingConfig
        self.config = config
        self.golden_source = golden_source or DefaultGoldenSource()
        self.is_running = False
        self._interval_minutes = getattr(config, "FORWARD_TEST_INTERVAL_MINUTES", {"5m": 5, "15m": 15, "1h": 60})
        _data_dir = getattr(config, "DATA_DIR", None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self._db_path = os.path.join(_data_dir, "model_registry.db")

    def _should_trigger(self, timeframe: str, now: datetime) -> bool:
        mins = self._interval_minutes.get(timeframe)
        if mins is None:
            return False
        # Trigger when minute % interval == 0 (e.g. :00, :05, :10 for 5m)
        return (now.minute % mins == 0) and (now.second < 30)

    def _tick(self) -> None:
        now = datetime.utcnow()
        for tf in self._interval_minutes:
            if not self._should_trigger(tf, now):
                continue
            campaigns = get_active_campaigns_for_timeframe(tf, db_path=self._db_path)
            for c in campaigns:
                try:
                    run_one_forward_test(
                        c["version_id"],
                        c["symbol"],
                        c["timeframe"],
                        golden_source=self.golden_source,
                        db_path=self._db_path,
                        config=self.config,
                    )
                except Exception as e:
                    logger.error(f"Forward test run failed {c}: {e}", exc_info=True)

    def run(self, tick_seconds: int = 60) -> None:
        """Run scheduler loop; check every tick_seconds (default 60)."""
        import time
        logger.info("Forward test scheduler started")
        self.is_running = True
        try:
            while self.is_running:
                self._tick()
                time.sleep(tick_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.is_running = False
            logger.info("Forward test scheduler stopped")

    def stop(self) -> None:
        self.is_running = False
