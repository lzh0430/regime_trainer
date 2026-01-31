"""
Forward testing: post-training hook, timeframe-based runs, golden source comparison, evidence tables.
"""
import os
import sqlite3
import logging
import schedule
import threading
import time
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
    cron_manager: Optional['ForwardTestCronManager'] = None,
) -> None:
    """
    Enroll (version_id, symbol, timeframe) for forward testing. Idempotent.
    If cron_manager is provided, registers a cron job for this campaign.
    """
    db_path = db_path or _default_db_path()
    _init_forward_test_tables(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    is_new = False
    with _get_conn(db_path) as conn:
        cur = conn.execute(
            "SELECT id FROM forward_test_campaigns WHERE version_id=? AND symbol=? AND timeframe=?",
            (version_id, symbol, timeframe),
        )
        existing = cur.fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO forward_test_campaigns
                (version_id, symbol, timeframe, created_at, required_runs, status)
                VALUES (?, ?, ?, ?, ?, 'active')
                """,
                (version_id, symbol, timeframe, now, required_runs),
            )
            conn.commit()
            is_new = True
    logger.info(f"Forward test campaign enrolled: {version_id} {symbol} {timeframe}")
    
    # Register cron job if manager is provided and this is a new campaign
    if cron_manager is not None and is_new:
        cron_manager.register_campaign_job(version_id, symbol, timeframe)


def on_training_finished(
    symbol: str,
    primary_timeframe: str,
    version_id: str,
    config: Any,
    db_path: Optional[str] = None,
    cron_manager: Optional['ForwardTestCronManager'] = None,
) -> None:
    """
    Hook to call at end of full_retrain / incremental_train. Enrolls campaign with required_runs from config.
    If cron_manager is provided, registers a cron job for this campaign.
    """
    required_runs = getattr(config, "FORWARD_TEST_REQUIRED_RUNS", 5)
    if db_path is None:
        base = getattr(config, "DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
        db_path = os.path.join(base, "model_registry.db")
    enroll_campaign(version_id, symbol, primary_timeframe, required_runs=required_runs, db_path=db_path, cron_manager=cron_manager)


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
            # Notify cron manager to cancel job (if using cron manager)
            try:
                cron_mgr = ForwardTestCronManager._instance
                if cron_mgr is not None:
                    cron_mgr.cancel_campaign_job(version_id, symbol, timeframe)
            except Exception:
                pass  # Cron manager may not be initialized

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


def get_all_pending_campaigns(
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return list of all active campaigns with run_count < required_runs (across all timeframes)."""
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return []
    _init_forward_test_tables(db_path)
    with _get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, version_id, symbol, timeframe, required_runs FROM forward_test_campaigns WHERE status = 'active'"
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


def trigger_all_pending_forward_tests(
    golden_source: Optional[GoldenSource] = None,
    db_path: Optional[str] = None,
    config: Any = None,
) -> Dict[str, Any]:
    """
    Trigger forward tests for all pending campaigns (active campaigns that still need runs).
    Returns summary dict with counts and results.
    """
    db_path = db_path or _default_db_path()
    campaigns = get_all_pending_campaigns(db_path=db_path)
    
    if not campaigns:
        logger.info("No pending forward test campaigns found")
        return {
            "total_campaigns": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "skipped_runs": 0,
            "results": [],
        }
    
    logger.info(f"Triggering forward tests for {len(campaigns)} pending campaigns")
    
    successful = 0
    failed = 0
    skipped = 0
    results = []
    
    for campaign in campaigns:
        version_id = campaign["version_id"]
        symbol = campaign["symbol"]
        timeframe = campaign["timeframe"]
        campaign_id = campaign["id"]
        
        try:
            result = run_one_forward_test(
                version_id,
                symbol,
                timeframe,
                golden_source=golden_source,
                db_path=db_path,
                config=config,
            )
            if result is None:
                skipped += 1
                results.append({
                    "campaign_id": campaign_id,
                    "version_id": version_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "skipped",
                    "reason": "Campaign already has required runs or is not active",
                })
            else:
                successful += 1
                results.append({
                    "campaign_id": campaign_id,
                    "version_id": version_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "success",
                    "run_at": result.get("run_at"),
                    "model_regime": result.get("model_regime"),
                    "golden_regime": result.get("golden_regime"),
                    "match": result.get("match"),
                    "runs_count": result.get("runs_count"),
                    "required_runs": result.get("required_runs"),
                })
        except Exception as e:
            failed += 1
            logger.error(f"Failed to trigger forward test for {version_id} {symbol} {timeframe}: {e}", exc_info=True)
            results.append({
                "campaign_id": campaign_id,
                "version_id": version_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "error",
                "error": str(e),
            })
    
    summary = {
        "total_campaigns": len(campaigns),
        "successful_runs": successful,
        "failed_runs": failed,
        "skipped_runs": skipped,
        "results": results,
    }
    
    logger.info(f"Forward test trigger complete: {successful} successful, {failed} failed, {skipped} skipped")
    return summary


# --------------- Cron Manager (per-campaign cron jobs) ---------------

class ForwardTestCronManager:
    """
    Manages cron jobs for forward test campaigns. Each campaign gets its own cron job
    that runs at the correct interval (5m/15m/1h). Jobs are registered when campaigns
    are enrolled and cancelled when campaigns complete or become inactive.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Any = None, golden_source: Optional[GoldenSource] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Any = None, golden_source: Optional[GoldenSource] = None):
        if self._initialized:
            return
        if config is None:
            from config import TrainingConfig
            config = TrainingConfig
        self.config = config
        self.golden_source = golden_source or DefaultGoldenSource()
        self._interval_minutes = getattr(config, "FORWARD_TEST_INTERVAL_MINUTES", {"5m": 5, "15m": 15, "1h": 60})
        _data_dir = getattr(config, "DATA_DIR", None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self._db_path = os.path.join(_data_dir, "model_registry.db")
        self._jobs = {}  # {(version_id, symbol, timeframe): schedule.Job}
        self._is_running = False
        self._scheduler_thread = None
        self._initialized = True
    
    def _get_campaign_key(self, version_id: str, symbol: str, timeframe: str) -> tuple:
        return (version_id, symbol, timeframe)
    
    def _job_wrapper(self, version_id: str, symbol: str, timeframe: str) -> None:
        """Wrapper for schedule job that runs one forward test."""
        try:
            result = run_one_forward_test(
                version_id,
                symbol,
                timeframe,
                golden_source=self.golden_source,
                db_path=self._db_path,
                config=self.config,
            )
            if result is None:
                # Campaign completed or inactive - cancel the job
                self.cancel_campaign_job(version_id, symbol, timeframe)
        except Exception as e:
            logger.error(f"Forward test cron job failed {version_id} {symbol} {timeframe}: {e}", exc_info=True)
    
    def register_campaign_job(self, version_id: str, symbol: str, timeframe: str) -> bool:
        """
        Register a cron job for a campaign. Returns True if registered, False if already exists.
        """
        key = self._get_campaign_key(version_id, symbol, timeframe)
        if key in self._jobs:
            logger.debug(f"Cron job already exists for {version_id} {symbol} {timeframe}")
            return False
        
        mins = self._interval_minutes.get(timeframe)
        if mins is None:
            logger.warning(f"Unknown timeframe {timeframe}, cannot register cron job")
            return False
        
        # Create schedule job: every N minutes
        job = schedule.every(mins).minutes.do(self._job_wrapper, version_id, symbol, timeframe)
        self._jobs[key] = job
        logger.info(f"Registered cron job for {version_id} {symbol} {timeframe}: every {mins} minutes")
        return True
    
    def cancel_campaign_job(self, version_id: str, symbol: str, timeframe: str) -> bool:
        """
        Cancel a cron job for a campaign. Returns True if cancelled, False if not found.
        """
        key = self._get_campaign_key(version_id, symbol, timeframe)
        if key not in self._jobs:
            return False
        
        schedule.cancel_job(self._jobs[key])
        del self._jobs[key]
        logger.info(f"Cancelled cron job for {version_id} {symbol} {timeframe}")
        return True
    
    def _scheduler_loop(self) -> None:
        """Background thread that runs schedule.run_pending() continuously."""
        logger.info("Forward test cron scheduler thread started")
        while self._is_running:
            try:
                schedule.run_pending()
                time.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Cron scheduler loop error: {e}", exc_info=True)
                time.sleep(1)
        logger.info("Forward test cron scheduler thread stopped")
    
    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._is_running:
            logger.warning("Cron manager already running")
            return
        self._is_running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Forward test cron manager started")
    
    def stop(self) -> None:
        """Stop the background scheduler thread and cancel all jobs."""
        if not self._is_running:
            return
        self._is_running = False
        # Cancel all jobs
        for key in list(self._jobs.keys()):
            schedule.cancel_job(self._jobs[key])
        self._jobs.clear()
        logger.info("Forward test cron manager stopped")
    
    def sync_jobs_from_db(self) -> None:
        """
        Sync cron jobs with DB: register jobs for active campaigns, cancel jobs for inactive/completed.
        """
        if not os.path.isfile(self._db_path):
            return
        _init_forward_test_tables(self._db_path)
        
        # Get all campaigns from DB
        with _get_conn(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT version_id, symbol, timeframe, status, required_runs FROM forward_test_campaigns"
            )
            db_campaigns = {tuple(row[:3]): (row[3], row[4]) for row in cur.fetchall()}
        
        # Register jobs for active campaigns that need more runs
        for (version_id, symbol, timeframe), (status, required_runs) in db_campaigns.items():
            if status != "active":
                self.cancel_campaign_job(version_id, symbol, timeframe)
                continue
            with _get_conn(self._db_path) as conn:
                cur = conn.execute(
                    "SELECT id FROM forward_test_campaigns WHERE version_id=? AND symbol=? AND timeframe=?",
                    (version_id, symbol, timeframe),
                )
                row = cur.fetchone()
                if row:
                    campaign_id = row[0]
                    n = _campaign_run_count(conn, campaign_id)
                    if n < required_runs:
                        self.register_campaign_job(version_id, symbol, timeframe)
                    else:
                        self.cancel_campaign_job(version_id, symbol, timeframe)
                else:
                    self.cancel_campaign_job(version_id, symbol, timeframe)
        
        # Cancel jobs for campaigns not in DB
        for key in list(self._jobs.keys()):
            if key not in db_campaigns:
                version_id, symbol, timeframe = key
                self.cancel_campaign_job(version_id, symbol, timeframe)
    
    def trigger_all_pending(self) -> Dict[str, Any]:
        """
        Trigger forward tests for all pending campaigns. Convenience method that uses
        the cron manager's config and golden_source.
        """
        return trigger_all_pending_forward_tests(
            golden_source=self.golden_source,
            db_path=self._db_path,
            config=self.config,
        )


# --------------- Scheduler (legacy polling-based) ---------------

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
