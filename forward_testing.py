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
                auto_promoted INTEGER DEFAULT 0,
                final_accuracy REAL,
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
        
        # Add new columns to existing table if they don't exist (migration)
        try:
            conn.execute("ALTER TABLE forward_test_campaigns ADD COLUMN auto_promoted INTEGER DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            conn.execute("ALTER TABLE forward_test_campaigns ADD COLUMN final_accuracy REAL")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists


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


def _calculate_campaign_accuracy(conn: sqlite3.Connection, campaign_id: int) -> Optional[Dict[str, Any]]:
    """
    Calculate accuracy metrics for a campaign.
    Returns dict with accuracy, total_runs, runs_with_golden, matches, or None if no runs.
    """
    cur = conn.execute(
        "SELECT match, golden_regime FROM forward_test_runs WHERE campaign_id = ?",
        (campaign_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    
    total_runs = len(rows)
    # Count runs that have golden source data (golden_regime is not None)
    runs_with_golden = sum(1 for row in rows if row[1] is not None)  # golden_regime is not None
    # Count matches (match == 1) - only count matches from runs with golden source
    matches = sum(1 for row in rows if row[1] is not None and row[0] == 1)  # golden_regime is not None and match == 1
    
    if runs_with_golden == 0:
        return {
            "accuracy": None,
            "total_runs": total_runs,
            "runs_with_golden": 0,
            "matches": 0,
            "has_sufficient_data": False,
        }
    
    accuracy = matches / runs_with_golden
    # Consider sufficient if at least 80% of runs have golden source data
    has_sufficient_data = runs_with_golden >= (total_runs * 0.8)
    
    return {
        "accuracy": accuracy,
        "total_runs": total_runs,
        "runs_with_golden": runs_with_golden,
        "matches": matches,
        "has_sufficient_data": has_sufficient_data,
    }


def _auto_promote_to_prod_if_qualified(
    version_id: str,
    symbol: str,
    timeframe: str,
    campaign_id: int,
    db_path: Optional[str] = None,
    config: Any = None,
    accuracy_threshold: float = 0.8,
    min_data_coverage: float = 0.8,
) -> bool:
    """
    Check if campaign meets auto-promotion criteria and promote to PROD if so.
    
    Criteria:
    - Accuracy >= accuracy_threshold (default 80%)
    - At least min_data_coverage (default 80%) of runs have golden source data
    
    Returns True if promoted, False otherwise.
    """
    db_path = db_path or _default_db_path()
    
    with _get_conn(db_path) as conn:
        accuracy_data = _calculate_campaign_accuracy(conn, campaign_id)
    
    if accuracy_data is None:
        logger.debug(f"Campaign {campaign_id} has no runs, cannot calculate accuracy")
        return False
    
    if accuracy_data["accuracy"] is None:
        logger.debug(f"Campaign {campaign_id} has no golden source data, cannot auto-promote")
        return False
    
    accuracy = accuracy_data["accuracy"]
    has_sufficient_data = accuracy_data["has_sufficient_data"]
    
    # Check if meets criteria
    if accuracy >= accuracy_threshold and has_sufficient_data:
        logger.info(
            f"Campaign {campaign_id} meets auto-promotion criteria: "
            f"accuracy={accuracy:.2%}, data_coverage={accuracy_data['runs_with_golden']}/{accuracy_data['total_runs']}"
        )
        
        # Import here to avoid circular dependency
        from model_registry import set_prod
        
        # Get models_dir from config
        if config is None:
            from config import TrainingConfig
            config = TrainingConfig
        models_dir = getattr(config, "MODELS_DIR", None)
        if models_dir is None:
            # Fallback to default models directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models")
        
        # Set PROD
        success = set_prod(symbol, timeframe, version_id, models_dir=models_dir, db_path=db_path)
        if success:
            # Update campaign record to mark as auto-promoted
            with _get_conn(db_path) as conn:
                conn.execute(
                    "UPDATE forward_test_campaigns SET auto_promoted = 1, final_accuracy = ? WHERE id = ?",
                    (accuracy, campaign_id),
                )
                conn.commit()
            
            logger.info(
                f"✅ Auto-promoted {version_id} {symbol} {timeframe} to PROD "
                f"(accuracy: {accuracy:.2%}, matches: {accuracy_data['matches']}/{accuracy_data['runs_with_golden']})"
            )
            return True
        else:
            logger.warning(
                f"Failed to auto-promote {version_id} {symbol} {timeframe} to PROD "
                f"(model path validation failed)"
            )
            return False
    
    # Update campaign record with final accuracy even if not promoted
    with _get_conn(db_path) as conn:
        conn.execute(
            "UPDATE forward_test_campaigns SET final_accuracy = ? WHERE id = ?",
            (accuracy, campaign_id),
        )
        conn.commit()
    
    logger.debug(
        f"Campaign {campaign_id} does not meet auto-promotion criteria: "
        f"accuracy={accuracy:.2%} (need >= {accuracy_threshold:.2%}), "
        f"data_coverage={accuracy_data['runs_with_golden']}/{accuracy_data['total_runs']} "
        f"(need >= {min_data_coverage:.2%})"
    )
    return False


def get_campaign_accuracy(
    version_id: str,
    symbol: str,
    timeframe: str,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get accuracy metrics for a forward test campaign.
    Returns dict with accuracy, total_runs, runs_with_golden, matches, etc., or None if campaign not found.
    """
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return None
    
    _init_forward_test_tables(db_path)
    
    with _get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id FROM forward_test_campaigns WHERE version_id = ? AND symbol = ? AND timeframe = ?",
            (version_id, symbol, timeframe),
        )
        row = cur.fetchone()
        if row is None:
            return None
        campaign_id = row["id"]
        
        accuracy_data = _calculate_campaign_accuracy(conn, campaign_id)
        if accuracy_data is None:
            return None
        
        # Get campaign status info
        cur = conn.execute(
            "SELECT status, auto_promoted, final_accuracy FROM forward_test_campaigns WHERE id = ?",
            (campaign_id,),
        )
        campaign_row = cur.fetchone()
        
        result = {
            "version_id": version_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "campaign_id": campaign_id,
            "status": campaign_row["status"] if campaign_row else None,
            "auto_promoted": bool(campaign_row["auto_promoted"]) if campaign_row and campaign_row["auto_promoted"] is not None else False,
            "final_accuracy": campaign_row["final_accuracy"] if campaign_row and campaign_row["final_accuracy"] is not None else None,
            **accuracy_data,
        }
        
        return result


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
            
            # Auto-promote to PROD if accuracy threshold met
            try:
                auto_promoted = _auto_promote_to_prod_if_qualified(
                    version_id, symbol, timeframe, campaign_id,
                    db_path=db_path, config=config,
                    accuracy_threshold=0.8,  # 80% accuracy required
                    min_data_coverage=0.8,  # 80% of runs must have golden source
                )
                if auto_promoted:
                    logger.info(f"✅ Model {version_id} {symbol} {timeframe} auto-promoted to PROD")
            except Exception as e:
                logger.warning(f"Auto-promotion hook failed (campaign still qualified): {e}", exc_info=True)

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
        logger.info(f"🔄 Cron job triggered: {version_id} {symbol} {timeframe}")
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
                logger.info(f"Campaign completed or inactive, cancelling cron job: {version_id} {symbol} {timeframe}")
                self.cancel_campaign_job(version_id, symbol, timeframe)
            else:
                logger.info(f"✅ Forward test run completed: {version_id} {symbol} {timeframe} - runs: {result.get('runs_count')}/{result.get('required_runs')}")
        except Exception as e:
            logger.error(f"❌ Forward test cron job failed {version_id} {symbol} {timeframe}: {e}", exc_info=True)
    
    def register_campaign_job(self, version_id: str, symbol: str, timeframe: str) -> bool:
        """
        Register a cron job for a campaign. Returns True if registered, False if already exists or invalid timeframe.
        """
        key = self._get_campaign_key(version_id, symbol, timeframe)
        if key in self._jobs:
            logger.debug(f"Cron job already exists for {version_id} {symbol} {timeframe}")
            return False
        
        mins = self._interval_minutes.get(timeframe)
        if mins is None:
            logger.warning(f"Unknown timeframe {timeframe}, cannot register cron job")
            return False
        
        # Ensure scheduler is running before registering jobs (only if timeframe is valid)
        if not self._is_running:
            logger.warning(f"Cron manager not running, starting it before registering job for {version_id} {symbol} {timeframe}")
            self.start()
        
        # Create schedule job: every N minutes
        job = schedule.every(mins).minutes.do(self._job_wrapper, version_id, symbol, timeframe)
        self._jobs[key] = job
        logger.info(f"✅ Registered cron job for {version_id} {symbol} {timeframe}: every {mins} minutes (total jobs: {len(self._jobs)})")
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
        tick_count = 0
        while self._is_running:
            try:
                schedule.run_pending()
                tick_count += 1
                # Log every 60 ticks (1 minute) to show scheduler is alive
                if tick_count % 60 == 0:
                    pending_jobs = len([j for j in schedule.jobs if j.should_run])
                    logger.debug(f"Cron scheduler alive: {len(self._jobs)} registered jobs, {pending_jobs} pending")
                # Sleep in small increments to check _is_running more frequently
                for _ in range(10):  # Sleep 0.1s at a time, total 1s
                    if not self._is_running:
                        break
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Cron scheduler loop error: {e}", exc_info=True)
                # On error, also check flag frequently
                for _ in range(10):
                    if not self._is_running:
                        break
                    time.sleep(0.1)
        logger.info("Forward test cron scheduler thread stopped")
    
    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._is_running:
            logger.warning("Cron manager already running")
            return
        self._is_running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="ForwardTestCronScheduler")
        self._scheduler_thread.start()
        logger.info(f"✅ Forward test cron manager started (thread: {self._scheduler_thread.name}, is_alive: {self._scheduler_thread.is_alive()})")
    
    def stop(self) -> None:
        """Stop the background scheduler thread and cancel all jobs."""
        if not self._is_running:
            return
        self._is_running = False
        # Cancel all jobs
        for key in list(self._jobs.keys()):
            try:
                schedule.cancel_job(self._jobs[key])
            except Exception:
                pass  # Job might already be cancelled
        self._jobs.clear()
        # Clear all schedule jobs to prevent interference
        try:
            schedule.clear()
        except Exception:
            pass  # schedule.clear() might fail in some versions
        logger.info("Forward test cron manager stopped")
    
    def sync_jobs_from_db(self) -> None:
        """
        Sync cron jobs with DB: register jobs for active campaigns, cancel jobs for inactive/completed.
        """
        if not os.path.isfile(self._db_path):
            logger.info("No database file found, skipping sync")
            return
        
        # Ensure scheduler is running
        if not self._is_running:
            logger.info("Starting cron manager before syncing jobs from DB")
            self.start()
        
        _init_forward_test_tables(self._db_path)
        
        # Get all campaigns from DB
        with _get_conn(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT version_id, symbol, timeframe, status, required_runs FROM forward_test_campaigns"
            )
            db_campaigns = {tuple(row[:3]): (row[3], row[4]) for row in cur.fetchall()}
        
        logger.info(f"Syncing cron jobs from DB: {len(db_campaigns)} campaigns found")
        
        registered_count = 0
        cancelled_count = 0
        
        # Register jobs for active campaigns that need more runs
        for (version_id, symbol, timeframe), (status, required_runs) in db_campaigns.items():
            if status != "active":
                if self.cancel_campaign_job(version_id, symbol, timeframe):
                    cancelled_count += 1
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
                        if self.register_campaign_job(version_id, symbol, timeframe):
                            registered_count += 1
                    else:
                        if self.cancel_campaign_job(version_id, symbol, timeframe):
                            cancelled_count += 1
                else:
                    if self.cancel_campaign_job(version_id, symbol, timeframe):
                        cancelled_count += 1
        
        # Cancel jobs for campaigns not in DB
        for key in list(self._jobs.keys()):
            if key not in db_campaigns:
                version_id, symbol, timeframe = key
                if self.cancel_campaign_job(version_id, symbol, timeframe):
                    cancelled_count += 1
        
        logger.info(f"✅ Sync complete: {registered_count} registered, {cancelled_count} cancelled, {len(self._jobs)} total jobs")
    
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
    
    def list_registered_jobs(self) -> List[Dict[str, Any]]:
        """List all currently registered cron jobs."""
        jobs = []
        for (version_id, symbol, timeframe), job in self._jobs.items():
            jobs.append({
                "version_id": version_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "interval_minutes": self._interval_minutes.get(timeframe),
                "next_run": str(job.next_run) if hasattr(job, 'next_run') else "unknown",
            })
        return jobs


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
