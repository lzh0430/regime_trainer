"""
Unit tests for forward testing hook (forward_testing.py).
Target: 100% coverage with temp dirs/DB and mocks for edge cases.

Run tests:
  python -m unittest test_forward_testing -v

Check coverage (requires: pip install coverage):
  python -m coverage run --source=forward_testing -m unittest test_forward_testing
  python -m coverage report -m
"""
import os
import sys
import sqlite3
import tempfile
import shutil
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forward_testing as ft

# Import schedule to clear jobs between tests
try:
    import schedule
except ImportError:
    schedule = None


def clear_schedule_jobs():
    """Helper to clear all schedule jobs, handling different schedule library versions."""
    if schedule is None:
        return
    try:
        # Try the clear() method (available in newer versions)
        schedule.clear()
    except AttributeError:
        # Fallback: manually clear jobs list
        try:
            schedule.jobs.clear()
        except AttributeError:
            # If jobs is not a list, try to cancel all jobs
            for job in list(schedule.jobs):
                try:
                    schedule.cancel_job(job)
                except Exception:
                    pass


class TestDefaultDbPath(unittest.TestCase):
    def test_default_db_path(self):
        out = ft._default_db_path()
        self.assertIn("data", out)
        self.assertTrue(out.endswith("model_registry.db"))


class TestGetConnAndInitTables(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "sub", "ft.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_conn_creates_parent_dir(self):
        conn = ft._get_conn(self.db)
        conn.close()
        self.assertTrue(os.path.isdir(os.path.dirname(self.db)))

    def test_init_forward_test_tables_creates_tables(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            r = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            names = {x[0] for x in r}
        self.assertIn("forward_test_campaigns", names)
        self.assertIn("forward_test_runs", names)

    def test_init_forward_test_tables_migration_handles_existing_columns(self):
        # Create table without new columns first - use _get_conn to ensure parent dir exists
        conn = ft._get_conn(self.db)
        with conn:
            conn.execute("""
                CREATE TABLE forward_test_campaigns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    required_runs INTEGER NOT NULL DEFAULT 5,
                    status TEXT NOT NULL DEFAULT 'active',
                    UNIQUE(version_id, symbol, timeframe)
                )
            """)
            conn.commit()
        # Now call init - should add columns without error
        ft._init_forward_test_tables(self.db)
        # Verify columns exist
        with sqlite3.connect(self.db) as c:
            cur = c.execute("PRAGMA table_info(forward_test_campaigns)")
            columns = [row[1] for row in cur.fetchall()]
        self.assertIn("auto_promoted", columns)
        self.assertIn("final_accuracy", columns)


class TestGoldenSource(unittest.TestCase):
    def test_default_golden_source_returns_none(self):
        gs = ft.DefaultGoldenSource()
        out = gs.get_golden_regime("BTCUSDT", "15m", datetime.utcnow())
        self.assertIsNone(out)


class TestEnrollCampaign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_enroll_campaign_creates_row(self):
        ft.enroll_campaign("v1", "BTCUSDT", "15m", required_runs=5, db_path=self.db)
        with sqlite3.connect(self.db) as c:
            r = c.execute(
                "SELECT version_id, symbol, timeframe, required_runs, status FROM forward_test_campaigns WHERE version_id=?",
                ("v1",),
            ).fetchone()
        self.assertEqual(r[0], "v1")
        self.assertEqual(r[1], "BTCUSDT")
        self.assertEqual(r[2], "15m")
        self.assertEqual(r[3], 5)
        self.assertEqual(r[4], "active")

    def test_enroll_campaign_idempotent(self):
        ft.enroll_campaign("v2", "ETHUSDT", "5m", required_runs=3, db_path=self.db)
        ft.enroll_campaign("v2", "ETHUSDT", "5m", required_runs=3, db_path=self.db)
        with sqlite3.connect(self.db) as c:
            n = c.execute("SELECT COUNT(*) FROM forward_test_campaigns WHERE version_id=? AND symbol=?", ("v2", "ETHUSDT")).fetchone()[0]
        self.assertEqual(n, 1)

    def test_enroll_campaign_uses_default_db_when_none(self):
        with patch.object(ft, "_default_db_path", return_value=self.db):
            ft.enroll_campaign("v3", "X", "1h", required_runs=5, db_path=None)
        self.assertTrue(os.path.isfile(self.db))

    def test_enroll_campaign_registers_cron_job_when_new_and_cron_manager_provided(self):
        mock_cron_mgr = MagicMock()
        mock_cron_mgr.register_campaign_job.return_value = True
        ft.enroll_campaign("v4", "BTC", "15m", required_runs=5, db_path=self.db, cron_manager=mock_cron_mgr)
        mock_cron_mgr.register_campaign_job.assert_called_once_with("v4", "BTC", "15m")

    def test_enroll_campaign_does_not_register_cron_job_when_existing(self):
        mock_cron_mgr = MagicMock()
        ft.enroll_campaign("v5", "BTC", "15m", required_runs=5, db_path=self.db, cron_manager=mock_cron_mgr)
        mock_cron_mgr.register_campaign_job.assert_called_once()
        # Second enrollment should not register again
        ft.enroll_campaign("v5", "BTC", "15m", required_runs=5, db_path=self.db, cron_manager=mock_cron_mgr)
        self.assertEqual(mock_cron_mgr.register_campaign_job.call_count, 1)


class TestOnTrainingFinished(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        self.config = MagicMock()
        self.config.FORWARD_TEST_REQUIRED_RUNS = 7
        self.config.DATA_DIR = os.path.dirname(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_on_training_finished_enrolls_with_config_required_runs(self):
        ft.on_training_finished("BTCUSDT", "15m", "v1", self.config, db_path=self.db)
        with sqlite3.connect(self.db) as c:
            r = c.execute("SELECT required_runs FROM forward_test_campaigns WHERE version_id=?", ("v1",)).fetchone()
        self.assertEqual(r[0], 7)

    def test_on_training_finished_uses_data_dir_when_db_path_none(self):
        ft.on_training_finished("ETHUSDT", "5m", "v2", self.config, db_path=None)
        expected_db = os.path.join(self.config.DATA_DIR, "model_registry.db")
        self.assertTrue(os.path.isfile(expected_db))

    def test_on_training_finished_passes_cron_manager(self):
        mock_cron_mgr = MagicMock()
        with patch("forward_testing.enroll_campaign") as mock_enroll:
            ft.on_training_finished("BTC", "15m", "v1", self.config, db_path=self.db, cron_manager=mock_cron_mgr)
        mock_enroll.assert_called_once()
        call_kwargs = mock_enroll.call_args[1]
        self.assertEqual(call_kwargs["cron_manager"], mock_cron_mgr)

    def test_on_training_finished_fallback_required_runs(self):
        config_no_attr = MagicMock(spec=[])  # no FORWARD_TEST_REQUIRED_RUNS
        config_no_attr.DATA_DIR = self.tmp
        db_path = os.path.join(self.tmp, "model_registry.db")
        ft.on_training_finished("X", "15m", "v3", config_no_attr, db_path=db_path)
        with sqlite3.connect(db_path) as c:
            r = c.execute("SELECT required_runs FROM forward_test_campaigns WHERE version_id=?", ("v3",)).fetchone()
        self.assertEqual(r[0], 5)


class TestCampaignRunCount(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_campaign_run_count(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, created_at) VALUES (1,?,?,?)",
                ("2025-01-01T00:00:00Z", "Range", "2025-01-01T00:00:00Z"),
            )
            c.commit()
            n = ft._campaign_run_count(c, 1)
        self.assertEqual(n, 1)


class TestRunOneForwardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        ft._init_forward_test_tables(self.db)
        self.config = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_one_forward_test_no_campaign_returns_none(self):
        mock_model_api = MagicMock()
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v_none", "BTCUSDT", "15m", db_path=self.db, config=self.config)
        self.assertIsNone(out)
        # ModelAPI is never instantiated when campaign is missing
        self.assertFalse(mock_model_api.ModelAPI.called)

    def test_run_one_forward_test_campaign_not_active_returns_none(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "qualified"),
            )
            c.commit()
        mock_model_api = MagicMock()
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=self.config)
        self.assertIsNone(out)

    def test_run_one_forward_test_already_required_runs_returns_none(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 1, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, created_at) VALUES (1,?,?,?)",
                ("2025-01-01T00:00:00Z", "Range", "2025-01-01T00:00:00Z"),
            )
            c.commit()
        mock_model_api = MagicMock()
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=self.config)
        self.assertIsNone(out)

    def test_run_one_forward_test_predict_raises_returns_none(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.side_effect = RuntimeError("mock")
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=self.config)
        self.assertIsNone(out)
        with sqlite3.connect(self.db) as c:
            n = c.execute("SELECT COUNT(*) FROM forward_test_runs WHERE campaign_id=1").fetchone()[0]
        self.assertEqual(n, 0)

    def test_run_one_forward_test_success_no_golden(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {
            "most_likely_regime": {"name": "Range", "id": 0, "probability": 0.9},
        }
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=self.config)
        self.assertIsNotNone(out)
        self.assertEqual(out["model_regime"], "Range")
        self.assertIsNone(out["golden_regime"])
        self.assertIsNone(out["match"])
        self.assertEqual(out["runs_count"], 1)
        self.assertEqual(out["required_runs"], 5)
        with sqlite3.connect(self.db) as c:
            r = c.execute("SELECT model_regime, golden_regime, match FROM forward_test_runs WHERE campaign_id=1").fetchone()
        self.assertEqual(r[0], "Range")
        self.assertIsNone(r[1])
        self.assertIsNone(r[2])

    def test_run_one_forward_test_success_golden_matches(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        golden = MagicMock()
        golden.get_golden_regime.return_value = "Range"
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v1", "BTC", "15m", golden_source=golden, db_path=self.db, config=self.config)
        self.assertEqual(out["match"], 1)
        with sqlite3.connect(self.db) as c:
            r = c.execute("SELECT match FROM forward_test_runs WHERE campaign_id=1").fetchone()
        self.assertEqual(r[0], 1)

    def test_run_one_forward_test_success_golden_mismatch(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        golden = MagicMock()
        golden.get_golden_regime.return_value = "Weak_Trend"
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v1", "BTC", "15m", golden_source=golden, db_path=self.db, config=self.config)
        self.assertEqual(out["match"], 0)

    def test_run_one_forward_test_success_qualifies_after_required_runs(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 2, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, created_at) VALUES (1,?,?,?)",
                ("2025-01-01T00:00:00Z", "Range", "2025-01-01T00:00:00Z"),
            )
            c.commit()
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=self.config)
        self.assertEqual(out["runs_count"], 2)
        with sqlite3.connect(self.db) as c:
            status = c.execute("SELECT status FROM forward_test_campaigns WHERE id=1").fetchone()[0]
        self.assertEqual(status, "qualified")

    def test_run_one_forward_test_config_none_uses_training_config(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            with patch("config.TrainingConfig"):
                out = ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=None)
        self.assertIsNotNone(out)

    def test_run_one_forward_test_golden_strip_comparison(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "  Range  "}}
        golden = MagicMock()
        golden.get_golden_regime.return_value = "Range"
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            out = ft.run_one_forward_test("v1", "BTC", "15m", golden_source=golden, db_path=self.db, config=self.config)
        self.assertEqual(out["match"], 1)


class TestGetActiveCampaignsForTimeframe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_active_campaigns_no_db_file_returns_empty(self):
        out = ft.get_active_campaigns_for_timeframe("15m", db_path=self.db)
        self.assertEqual(out, [])

    def test_get_active_campaigns_no_campaigns_returns_empty(self):
        ft._init_forward_test_tables(self.db)
        Path(self.db).touch()
        out = ft.get_active_campaigns_for_timeframe("15m", db_path=self.db)
        self.assertEqual(out, [])

    def test_get_active_campaigns_includes_only_run_count_less_than_required(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v2", "ETH", "15m", "2025-01-01T00:00:00Z", 1, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, created_at) VALUES (2,?,?,?)",
                ("2025-01-01T00:00:00Z", "Range", "2025-01-01T00:00:00Z"),
            )
            c.commit()
        out = ft.get_active_campaigns_for_timeframe("15m", db_path=self.db)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["version_id"], "v1")
        self.assertEqual(out[0]["run_count"], 0)


class TestForwardTestScheduler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        self.config = MagicMock()
        self.config.FORWARD_TEST_INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60}
        self.config.DATA_DIR = os.path.dirname(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scheduler_init_config_none(self):
        with patch("config.TrainingConfig") as TC:
            TC.FORWARD_TEST_INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60}
            TC.DATA_DIR = self.tmp
            s = ft.ForwardTestScheduler(config=None)
        self.assertIsInstance(s.golden_source, ft.DefaultGoldenSource)
        self.assertEqual(s._interval_minutes, {"5m": 5, "15m": 15, "1h": 60})

    def test_scheduler_init_with_config_and_golden_source(self):
        golden = ft.DefaultGoldenSource()
        s = ft.ForwardTestScheduler(config=self.config, golden_source=golden)
        self.assertIs(s.golden_source, golden)
        self.assertEqual(s._interval_minutes, {"5m": 5, "15m": 15, "1h": 60})
        self.assertEqual(s._db_path, os.path.join(self.config.DATA_DIR, "model_registry.db"))

    def test_scheduler_init_config_without_data_dir_uses_fallback(self):
        config_no_data = MagicMock()
        config_no_data.FORWARD_TEST_INTERVAL_MINUTES = {"5m": 5}
        config_no_data.DATA_DIR = None
        s = ft.ForwardTestScheduler(config=config_no_data)
        self.assertIn("data", s._db_path)
        self.assertTrue(s._db_path.endswith("model_registry.db"))

    def test_should_trigger_timeframe_not_in_interval_returns_false(self):
        s = ft.ForwardTestScheduler(config=self.config)
        now = datetime(2025, 1, 1, 12, 0, 0)
        self.assertFalse(s._should_trigger("99m", now))

    def test_should_trigger_minute_not_aligned_returns_false(self):
        s = ft.ForwardTestScheduler(config=self.config)
        now = datetime(2025, 1, 1, 12, 7, 0)  # 7 % 5 != 0
        self.assertFalse(s._should_trigger("5m", now))

    def test_should_trigger_second_ge_30_returns_false(self):
        s = ft.ForwardTestScheduler(config=self.config)
        now = datetime(2025, 1, 1, 12, 0, 30)
        self.assertFalse(s._should_trigger("5m", now))

    def test_should_trigger_aligned_returns_true(self):
        s = ft.ForwardTestScheduler(config=self.config)
        now = datetime(2025, 1, 1, 12, 0, 10)
        self.assertTrue(s._should_trigger("5m", now))
        now15 = datetime(2025, 1, 1, 12, 15, 5)
        self.assertTrue(s._should_trigger("15m", now15))
        now1h = datetime(2025, 1, 1, 12, 0, 0)
        self.assertTrue(s._should_trigger("1h", now1h))

    def test_tick_calls_run_one_forward_test_for_each_active_campaign(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        s = ft.ForwardTestScheduler(config=self.config)
        s._db_path = self.db
        with patch("forward_testing.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2025, 1, 1, 12, 15, 10)
            with patch("forward_testing.run_one_forward_test") as mock_run:
                s._tick()
        mock_run.assert_called_once()
        call_kw = mock_run.call_args[1]
        self.assertEqual(call_kw["db_path"], self.db)
        self.assertEqual(call_kw["config"], self.config)

    def test_tick_handles_run_exception(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        s = ft.ForwardTestScheduler(config=self.config)
        s._db_path = self.db
        with patch("forward_testing.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2025, 1, 1, 12, 15, 10)
            with patch("forward_testing.run_one_forward_test", side_effect=RuntimeError("mock")):
                s._tick()
        # should not raise

    def test_run_stops_when_is_running_false(self):
        s = ft.ForwardTestScheduler(config=self.config)
        s._db_path = self.db
        call_count = 0

        def fake_sleep(secs):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                s.stop()

        with patch("time.sleep", side_effect=fake_sleep):
            with patch.object(s, "_tick"):
                s.run(tick_seconds=60)
        self.assertFalse(s.is_running)

    def test_stop_sets_is_running_false(self):
        s = ft.ForwardTestScheduler(config=self.config)
        s.is_running = True
        s.stop()
        self.assertFalse(s.is_running)

    def test_run_handles_keyboard_interrupt(self):
        s = ft.ForwardTestScheduler(config=self.config)
        s._db_path = self.db

        def fake_sleep_raise(_secs):
            raise KeyboardInterrupt()

        with patch.object(s, "_tick"):
            with patch("time.sleep", side_effect=fake_sleep_raise):
                s.run(tick_seconds=60)
        self.assertFalse(s.is_running)


class TestCalculateCampaignAccuracy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        ft._init_forward_test_tables(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_calculate_campaign_accuracy_no_runs_returns_none(self):
        with sqlite3.connect(self.db) as conn:
            result = ft._calculate_campaign_accuracy(conn, campaign_id=999)
        self.assertIsNone(result)

    def test_calculate_campaign_accuracy_no_golden_source_returns_none_accuracy(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:00:00Z", "Range", None, None, "2025-01-01T00:00:00Z"),
            )
            c.commit()
        with sqlite3.connect(self.db) as conn:
            result = ft._calculate_campaign_accuracy(conn, campaign_id=1)
        self.assertIsNotNone(result)
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["total_runs"], 1)
        self.assertEqual(result["runs_with_golden"], 0)
        self.assertEqual(result["matches"], 0)
        self.assertFalse(result["has_sufficient_data"])

    def test_calculate_campaign_accuracy_all_matches(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            for i in range(5):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            c.commit()
        with sqlite3.connect(self.db) as conn:
            result = ft._calculate_campaign_accuracy(conn, campaign_id=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["total_runs"], 5)
        self.assertEqual(result["runs_with_golden"], 5)
        self.assertEqual(result["matches"], 5)
        self.assertTrue(result["has_sufficient_data"])

    def test_calculate_campaign_accuracy_partial_matches(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # 4 matches out of 5 runs with golden source
            for i in range(4):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:04:00Z", "Range", "Trend", 0, "2025-01-01T00:04:00Z"),
            )
            c.commit()
        with sqlite3.connect(self.db) as conn:
            result = ft._calculate_campaign_accuracy(conn, campaign_id=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["accuracy"], 0.8)  # 4/5 = 0.8
        self.assertEqual(result["total_runs"], 5)
        self.assertEqual(result["runs_with_golden"], 5)
        self.assertEqual(result["matches"], 4)
        self.assertTrue(result["has_sufficient_data"])

    def test_calculate_campaign_accuracy_insufficient_data_coverage(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # Only 3 out of 5 runs have golden source (< 80%)
            for i in range(3):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            for i in range(3, 5):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", None, None, f"2025-01-01T00:0{i}:00Z"),
                )
            c.commit()
        with sqlite3.connect(self.db) as conn:
            result = ft._calculate_campaign_accuracy(conn, campaign_id=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["accuracy"], 1.0)  # 3/3 = 1.0
        self.assertEqual(result["total_runs"], 5)
        self.assertEqual(result["runs_with_golden"], 3)
        self.assertEqual(result["matches"], 3)
        self.assertFalse(result["has_sufficient_data"])  # 3/5 = 60% < 80%

    def test_calculate_campaign_accuracy_mixed_runs(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # Mix: 2 matches, 1 mismatch, 2 no golden source
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:00:00Z", "Range", "Range", 1, "2025-01-01T00:00:00Z"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:01:00Z", "Range", "Range", 1, "2025-01-01T00:01:00Z"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:02:00Z", "Range", "Trend", 0, "2025-01-01T00:02:00Z"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:03:00Z", "Range", None, None, "2025-01-01T00:03:00Z"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:04:00Z", "Range", None, None, "2025-01-01T00:04:00Z"),
            )
            c.commit()
        with sqlite3.connect(self.db) as conn:
            result = ft._calculate_campaign_accuracy(conn, campaign_id=1)
        self.assertIsNotNone(result)
        self.assertEqual(result["accuracy"], 2.0 / 3.0)  # 2 matches / 3 runs with golden
        self.assertEqual(result["total_runs"], 5)
        self.assertEqual(result["runs_with_golden"], 3)
        self.assertEqual(result["matches"], 2)
        self.assertFalse(result["has_sufficient_data"])  # 3/5 = 60% < 80%


class TestAutoPromoteToProdIfQualified(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        self.models_dir = os.path.join(self.tmp, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        ft._init_forward_test_tables(self.db)
        self.config = MagicMock()
        self.config.MODELS_DIR = self.models_dir

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_auto_promote_no_runs_returns_false(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        result = ft._auto_promote_to_prod_if_qualified("v1", "BTC", "15m", 1, db_path=self.db, config=self.config)
        self.assertFalse(result)

    def test_auto_promote_no_golden_source_returns_false(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            for i in range(5):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", None, None, f"2025-01-01T00:0{i}:00Z"),
                )
            c.commit()
        result = ft._auto_promote_to_prod_if_qualified("v1", "BTC", "15m", 1, db_path=self.db, config=self.config)
        self.assertFalse(result)

    def test_auto_promote_accuracy_below_threshold_returns_false(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # 3 matches out of 5 = 60% accuracy (< 80%)
            for i in range(3):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            for i in range(3, 5):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Trend", 0, f"2025-01-01T00:0{i}:00Z"),
                )
            c.commit()
        
        with patch("model_registry.set_prod") as mock_set_prod:
            result = ft._auto_promote_to_prod_if_qualified("v1", "BTC", "15m", 1, db_path=self.db, config=self.config)
        self.assertFalse(result)
        mock_set_prod.assert_not_called()
        
        # Check final_accuracy was saved
        with sqlite3.connect(self.db) as c:
            row = c.execute("SELECT final_accuracy FROM forward_test_campaigns WHERE id=1").fetchone()
            self.assertIsNotNone(row[0])
            self.assertEqual(row[0], 0.6)  # 3/5 = 0.6

    def test_auto_promote_insufficient_data_coverage_returns_false(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # Only 3 out of 5 runs have golden source (60% < 80% coverage)
            for i in range(3):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            for i in range(3, 5):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", None, None, f"2025-01-01T00:0{i}:00Z"),
                )
            c.commit()
        
        with patch("model_registry.set_prod") as mock_set_prod:
            result = ft._auto_promote_to_prod_if_qualified("v1", "BTC", "15m", 1, db_path=self.db, config=self.config)
        self.assertFalse(result)
        mock_set_prod.assert_not_called()

    def test_auto_promote_meets_criteria_promotes_successfully(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # 4 matches out of 5 = 80% accuracy, all 5 have golden source = 100% coverage
            for i in range(4):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:04:00Z", "Range", "Trend", 0, "2025-01-01T00:04:00Z"),
            )
            c.commit()
        
        with patch("model_registry.set_prod", return_value=True) as mock_set_prod:
            result = ft._auto_promote_to_prod_if_qualified("v1", "BTC", "15m", 1, db_path=self.db, config=self.config)
        self.assertTrue(result)
        mock_set_prod.assert_called_once_with("BTC", "15m", "v1", models_dir=self.models_dir, db_path=self.db)
        
        # Check campaign was marked as auto-promoted
        with sqlite3.connect(self.db) as c:
            row = c.execute("SELECT auto_promoted, final_accuracy FROM forward_test_campaigns WHERE id=1").fetchone()
            self.assertEqual(row[0], 1)
            self.assertEqual(row[1], 0.8)  # 4/5 = 0.8

    def test_auto_promote_set_prod_fails_returns_false(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # 5 matches out of 5 = 100% accuracy
            for i in range(5):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            c.commit()
        
        with patch("model_registry.set_prod", return_value=False) as mock_set_prod:
            result = ft._auto_promote_to_prod_if_qualified("v1", "BTC", "15m", 1, db_path=self.db, config=self.config)
        self.assertFalse(result)
        mock_set_prod.assert_called_once()
        
        # Check campaign was NOT marked as auto-promoted
        with sqlite3.connect(self.db) as c:
            row = c.execute("SELECT auto_promoted FROM forward_test_campaigns WHERE id=1").fetchone()
            self.assertEqual(row[0], 0)

    def test_auto_promote_custom_thresholds(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # 3 matches out of 5 = 60% accuracy
            for i in range(3):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            for i in range(3, 5):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Trend", 0, f"2025-01-01T00:0{i}:00Z"),
                )
            c.commit()
        
        # Use lower threshold (50%)
        with patch("model_registry.set_prod", return_value=True) as mock_set_prod:
            result = ft._auto_promote_to_prod_if_qualified(
                "v1", "BTC", "15m", 1,
                db_path=self.db, config=self.config,
                accuracy_threshold=0.5, min_data_coverage=0.8
            )
        self.assertTrue(result)
        mock_set_prod.assert_called_once()

    def test_auto_promote_config_none_uses_training_config(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # 4 matches out of 5 = 80% accuracy
            for i in range(4):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:04:00Z", "Range", "Trend", 0, "2025-01-01T00:04:00Z"),
            )
            c.commit()
        
        mock_config = MagicMock()
        mock_config.MODELS_DIR = self.models_dir
        
        with patch("config.TrainingConfig", mock_config):
            with patch("model_registry.set_prod", return_value=True) as mock_set_prod:
                result = ft._auto_promote_to_prod_if_qualified("v1", "BTC", "15m", 1, db_path=self.db, config=None)
        self.assertTrue(result)
        mock_set_prod.assert_called_once()


class TestGetCampaignAccuracy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        ft._init_forward_test_tables(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_campaign_accuracy_no_db_returns_none(self):
        result = ft.get_campaign_accuracy("v1", "BTC", "15m", db_path=os.path.join(self.tmp, "nonexistent.db"))
        self.assertIsNone(result)

    def test_get_campaign_accuracy_campaign_not_found_returns_none(self):
        result = ft.get_campaign_accuracy("v1", "BTC", "15m", db_path=self.db)
        self.assertIsNone(result)

    def test_get_campaign_accuracy_no_runs_returns_none(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        result = ft.get_campaign_accuracy("v1", "BTC", "15m", db_path=self.db)
        self.assertIsNone(result)

    def test_get_campaign_accuracy_returns_complete_metrics(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status, auto_promoted, final_accuracy) VALUES (?,?,?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "qualified", 1, 0.8),
            )
            # 4 matches out of 5
            for i in range(4):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:04:00Z", "Range", "Trend", 0, "2025-01-01T00:04:00Z"),
            )
            c.commit()
        result = ft.get_campaign_accuracy("v1", "BTC", "15m", db_path=self.db)
        self.assertIsNotNone(result)
        self.assertEqual(result["version_id"], "v1")
        self.assertEqual(result["symbol"], "BTC")
        self.assertEqual(result["timeframe"], "15m")
        self.assertEqual(result["campaign_id"], 1)
        self.assertEqual(result["status"], "qualified")
        self.assertTrue(result["auto_promoted"])
        self.assertEqual(result["final_accuracy"], 0.8)
        self.assertEqual(result["accuracy"], 0.8)
        self.assertEqual(result["total_runs"], 5)
        self.assertEqual(result["runs_with_golden"], 5)
        self.assertEqual(result["matches"], 4)
        self.assertTrue(result["has_sufficient_data"])


class TestAutoPromotionHookIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        self.models_dir = os.path.join(self.tmp, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        ft._init_forward_test_tables(self.db)
        self.config = MagicMock()
        self.config.MODELS_DIR = self.models_dir

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_one_forward_test_auto_promotes_when_qualified_and_meets_criteria(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 2, "active"),
            )
            # Already 1 run with match
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:00:00Z", "Range", "Range", 1, "2025-01-01T00:00:00Z"),
            )
            c.commit()
        
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        
        golden = MagicMock()
        golden.get_golden_regime.return_value = "Range"  # Match
        
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            with patch("model_registry.set_prod", return_value=True) as mock_set_prod:
                with patch.object(ft, "ForwardTestCronManager") as mock_cron_mgr:
                    mock_cron_mgr._instance = None
                    result = ft.run_one_forward_test(
                        "v1", "BTC", "15m",
                        golden_source=golden,
                        db_path=self.db,
                        config=self.config
                    )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["runs_count"], 2)
        
        # Check campaign was qualified
        with sqlite3.connect(self.db) as c:
            row = c.execute("SELECT status, auto_promoted, final_accuracy FROM forward_test_campaigns WHERE id=1").fetchone()
            self.assertEqual(row[0], "qualified")
            # Should be auto-promoted (2 matches / 2 runs = 100% accuracy)
            self.assertEqual(row[1], 1)
            self.assertEqual(row[2], 1.0)
        
        # Check set_prod was called
        mock_set_prod.assert_called_once_with("BTC", "15m", "v1", models_dir=self.models_dir, db_path=self.db)

    def test_run_one_forward_test_no_auto_promote_when_accuracy_below_threshold(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 2, "active"),
            )
            # Already 1 run with mismatch
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:00:00Z", "Range", "Trend", 0, "2025-01-01T00:00:00Z"),
            )
            c.commit()
        
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        
        golden = MagicMock()
        golden.get_golden_regime.return_value = "Trend"  # Mismatch
        
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            with patch("model_registry.set_prod") as mock_set_prod:
                with patch.object(ft, "ForwardTestCronManager") as mock_cron_mgr:
                    mock_cron_mgr._instance = None
                    result = ft.run_one_forward_test(
                        "v1", "BTC", "15m",
                        golden_source=golden,
                        db_path=self.db,
                        config=self.config
                    )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["runs_count"], 2)
        
        # Check campaign was qualified but NOT auto-promoted (0 matches / 2 runs = 0% accuracy)
        with sqlite3.connect(self.db) as c:
            row = c.execute("SELECT status, auto_promoted, final_accuracy FROM forward_test_campaigns WHERE id=1").fetchone()
            self.assertEqual(row[0], "qualified")
            self.assertEqual(row[1], 0)  # Not auto-promoted
            self.assertEqual(row[2], 0.0)  # 0/2 = 0.0
        
        # Check set_prod was NOT called
        mock_set_prod.assert_not_called()

    def test_run_one_forward_test_cancels_cron_job_when_qualified(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 1, "active"),
            )
            c.commit()
        
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        
        mock_cron_mgr = MagicMock()
        mock_cron_mgr.cancel_campaign_job.return_value = True
        
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            with patch.object(ft.ForwardTestCronManager, "_instance", mock_cron_mgr):
                ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=self.config)
        
        mock_cron_mgr.cancel_campaign_job.assert_called_once_with("v1", "BTC", "15m")

    def test_run_one_forward_test_cron_manager_not_initialized_handles_gracefully(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 1, "active"),
            )
            c.commit()
        
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        
        # Set _instance to None to simulate not initialized
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            with patch.object(ft.ForwardTestCronManager, "_instance", None):
                # Should not raise exception
                result = ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=self.config)
                self.assertIsNotNone(result)

    def test_run_one_forward_test_config_none_uses_training_config(self):
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 1, "active"),
            )
            c.commit()
        
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        
        mock_tc = MagicMock()
        
        with patch.dict("sys.modules", {"model_api": mock_model_api, "config": MagicMock(TrainingConfig=mock_tc)}):
            with patch("forward_testing.TrainingConfig", mock_tc, create=True):
                result = ft.run_one_forward_test("v1", "BTC", "15m", db_path=self.db, config=None)
        
        self.assertIsNotNone(result)
        mock_model_api.ModelAPI.assert_called_once_with(mock_tc)

    def test_run_one_forward_test_auto_promotion_hook_failure_does_not_affect_qualification(self):
        # Create model directory
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 2, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:00:00Z", "Range", "Range", 1, "2025-01-01T00:00:00Z"),
            )
            c.commit()
        
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        
        golden = MagicMock()
        golden.get_golden_regime.return_value = "Range"
        
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            with patch("model_registry.set_prod", side_effect=Exception("Mock error")) as mock_set_prod:
                with patch.object(ft, "ForwardTestCronManager") as mock_cron_mgr:
                    mock_cron_mgr._instance = None
                    result = ft.run_one_forward_test(
                        "v1", "BTC", "15m",
                        golden_source=golden,
                        db_path=self.db,
                        config=self.config
                    )
        
        # Campaign should still be qualified even if auto-promotion fails
        self.assertIsNotNone(result)
        self.assertEqual(result["runs_count"], 2)
        
        with sqlite3.connect(self.db) as c:
            row = c.execute("SELECT status FROM forward_test_campaigns WHERE id=1").fetchone()
            self.assertEqual(row[0], "qualified")


class TestGetAllPendingCampaigns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_all_pending_campaigns_no_db_returns_empty(self):
        result = ft.get_all_pending_campaigns(db_path=self.db)
        self.assertEqual(result, [])

    def test_get_all_pending_campaigns_no_campaigns_returns_empty(self):
        ft._init_forward_test_tables(self.db)
        Path(self.db).touch()
        result = ft.get_all_pending_campaigns(db_path=self.db)
        self.assertEqual(result, [])

    def test_get_all_pending_campaigns_returns_only_pending(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v2", "ETH", "5m", "2025-01-01T00:00:00Z", 2, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v3", "SOL", "1h", "2025-01-01T00:00:00Z", 1, "active"),
            )
            # v2 has 1 run (needs 2)
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, created_at) VALUES (?,?,?,?)",
                (2, "2025-01-01T00:00:00Z", "Range", "2025-01-01T00:00:00Z"),
            )
            # v3 has 1 run (has 1, so not pending)
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, created_at) VALUES (?,?,?,?)",
                (3, "2025-01-01T00:00:00Z", "Range", "2025-01-01T00:00:00Z"),
            )
            c.commit()
        result = ft.get_all_pending_campaigns(db_path=self.db)
        self.assertEqual(len(result), 2)  # v1 and v2
        version_ids = {r["version_id"] for r in result}
        self.assertIn("v1", version_ids)
        self.assertIn("v2", version_ids)
        self.assertNotIn("v3", version_ids)


class TestTriggerAllPendingForwardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        self.config = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_trigger_all_pending_no_campaigns_returns_empty_summary(self):
        result = ft.trigger_all_pending_forward_tests(db_path=self.db, config=self.config)
        self.assertEqual(result["total_campaigns"], 0)
        self.assertEqual(result["successful_runs"], 0)
        self.assertEqual(result["failed_runs"], 0)
        self.assertEqual(result["skipped_runs"], 0)
        self.assertEqual(result["results"], [])

    def test_trigger_all_pending_successful_runs(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            result = ft.trigger_all_pending_forward_tests(db_path=self.db, config=self.config)
        
        self.assertEqual(result["total_campaigns"], 1)
        self.assertEqual(result["successful_runs"], 1)
        self.assertEqual(result["failed_runs"], 0)
        self.assertEqual(result["skipped_runs"], 0)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["status"], "success")

    def test_trigger_all_pending_skipped_when_already_has_runs(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 1, "active"),  # required_runs=1
            )
            # Already has 1 run (equals required_runs), so should be filtered out
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, created_at) VALUES (?,?,?,?)",
                (1, "2025-01-01T00:00:00Z", "Range", "2025-01-01T00:00:00Z"),
            )
            c.commit()
        
        mock_api = MagicMock()
        mock_api.predict_next_regime_for_version.return_value = {"most_likely_regime": {"name": "Range"}}
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.return_value = mock_api
        
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            result = ft.trigger_all_pending_forward_tests(db_path=self.db, config=self.config)
        
        # Campaign has 1 run and required_runs=1, so it won't be in pending list
        # (filtered out by get_all_pending_campaigns because 1 is not < 1)
        # This is correct behavior - campaigns with enough runs are not pending
        self.assertEqual(result["total_campaigns"], 0)
        self.assertEqual(result["successful_runs"], 0)
        self.assertEqual(result["skipped_runs"], 0)
        self.assertEqual(len(result["results"]), 0)

    def test_trigger_all_pending_handles_exceptions(self):
        ft._init_forward_test_tables(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        
        mock_model_api = MagicMock()
        mock_model_api.ModelAPI.side_effect = Exception("API error")
        
        with patch.dict("sys.modules", {"model_api": mock_model_api}):
            result = ft.trigger_all_pending_forward_tests(db_path=self.db, config=self.config)
        
        self.assertEqual(result["total_campaigns"], 1)
        self.assertEqual(result["failed_runs"], 1)
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertIn("error", result["results"][0])


class TestForwardTestCronManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        self.models_dir = os.path.join(self.tmp, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.config = MagicMock()
        self.config.FORWARD_TEST_INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60}
        self.config.DATA_DIR = os.path.dirname(self.db)
        # Reset singleton
        ft.ForwardTestCronManager._instance = None

    def tearDown(self):
        # Stop scheduler if running - must do this before resetting singleton
        if ft.ForwardTestCronManager._instance is not None:
            instance = ft.ForwardTestCronManager._instance
            try:
                if instance._is_running or (instance._scheduler_thread is not None and instance._scheduler_thread.is_alive()):
                    # Set flag to stop the loop first
                    instance._is_running = False
                    # Cancel all jobs and clear schedule
                    instance.stop()
                    # Wait for thread to finish (with timeout)
                    if instance._scheduler_thread is not None and instance._scheduler_thread.is_alive():
                        instance._scheduler_thread.join(timeout=2.0)
                        # If still alive after timeout, log but continue (daemon thread will be killed)
                        if instance._scheduler_thread.is_alive():
                            logger.warning("Scheduler thread did not stop within timeout, but continuing cleanup")
            except Exception as e:
                logger.debug(f"Error stopping scheduler in tearDown: {e}")
                pass  # Ignore errors during cleanup
        # Clear schedule jobs to prevent interference between tests
        clear_schedule_jobs()
        shutil.rmtree(self.tmp, ignore_errors=True)
        # Reset singleton after stopping scheduler
        ft.ForwardTestCronManager._instance = None

    def test_cron_manager_singleton(self):
        mgr1 = ft.ForwardTestCronManager(self.config)
        mgr2 = ft.ForwardTestCronManager(self.config)
        self.assertIs(mgr1, mgr2)

    def test_cron_manager_init_with_config_none(self):
        with patch("config.TrainingConfig") as mock_tc:
            mock_tc.FORWARD_TEST_INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60}
            mock_tc.DATA_DIR = None
            mgr = ft.ForwardTestCronManager(config=None)
            self.assertEqual(mgr.config, mock_tc)

    def test_cron_manager_init_with_golden_source(self):
        mock_golden = MagicMock()
        mgr = ft.ForwardTestCronManager(self.config, golden_source=mock_golden)
        self.assertEqual(mgr.golden_source, mock_golden)

    def test_cron_manager_init_already_initialized_returns_early(self):
        mgr1 = ft.ForwardTestCronManager(self.config)
        original_init = mgr1._initialized
        mgr1._initialized = True
        
        # Second init should return early
        mgr2 = ft.ForwardTestCronManager(self.config)
        self.assertIs(mgr1, mgr2)
        
        # Restore for cleanup
        mgr1._initialized = original_init

    def test_cron_manager_get_campaign_key(self):
        mgr = ft.ForwardTestCronManager(self.config)
        key = mgr._get_campaign_key("v1", "BTC", "15m")
        self.assertEqual(key, ("v1", "BTC", "15m"))

    def test_register_campaign_job_already_exists_returns_false(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        mgr.start()
        
        with patch("schedule.every") as mock_every:
            mock_job = MagicMock()
            mock_every.return_value.minutes.do.return_value = mock_job
            result1 = mgr.register_campaign_job("v1", "BTC", "15m")
            result2 = mgr.register_campaign_job("v1", "BTC", "15m")
        
        self.assertTrue(result1)
        self.assertFalse(result2)
        mgr.stop()

    def test_register_campaign_job_unknown_timeframe_returns_false(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        result = mgr.register_campaign_job("v1", "BTC", "unknown")
        self.assertFalse(result)

    def test_register_campaign_job_starts_scheduler_if_not_running(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        self.assertFalse(mgr._is_running)
        
        with patch("schedule.every") as mock_every:
            mock_job = MagicMock()
            mock_every.return_value.minutes.do.return_value = mock_job
            mgr.register_campaign_job("v1", "BTC", "15m")
        
        self.assertTrue(mgr._is_running)
        mgr.stop()

    def test_cancel_campaign_job_not_found_returns_false(self):
        mgr = ft.ForwardTestCronManager(self.config)
        result = mgr.cancel_campaign_job("v1", "BTC", "15m")
        self.assertFalse(result)

    def test_cancel_campaign_job_success(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        mgr.start()
        
        with patch("schedule.every") as mock_every:
            mock_job = MagicMock()
            mock_every.return_value.minutes.do.return_value = mock_job
            mgr.register_campaign_job("v1", "BTC", "15m")
            self.assertEqual(len(mgr._jobs), 1)
            
            with patch("schedule.cancel_job") as mock_cancel:
                result = mgr.cancel_campaign_job("v1", "BTC", "15m")
        
        self.assertTrue(result)
        self.assertEqual(len(mgr._jobs), 0)
        mock_cancel.assert_called_once_with(mock_job)
        mgr.stop()

    def test_start_already_running_does_nothing(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        mgr._is_running = True
        
        with patch("threading.Thread") as mock_thread:
            mgr.start()
        
        mock_thread.assert_not_called()

    def test_stop_not_running_does_nothing(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._is_running = False
        mgr.stop()  # Should not raise error

    def test_stop_cancels_all_jobs(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        mgr._is_running = True  # Must be running for stop() to cancel jobs
        
        mock_job1 = MagicMock()
        mock_job2 = MagicMock()
        mgr._jobs = {("v1", "BTC", "15m"): mock_job1, ("v2", "ETH", "5m"): mock_job2}
        
        with patch("schedule.cancel_job") as mock_cancel:
            with patch("schedule.clear") as mock_clear:  # Mock clear to avoid clearing before cancel
                mgr.stop()
        
        self.assertEqual(mock_cancel.call_count, 2)
        self.assertEqual(len(mgr._jobs), 0)

    def test_sync_jobs_from_db_no_db_file(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = os.path.join(self.tmp, "nonexistent.db")
        mgr.sync_jobs_from_db()  # Should not raise error
        # Ensure scheduler is not started when no DB file
        self.assertFalse(mgr._is_running)

    def test_sync_jobs_from_db_handles_unknown_timeframe_gracefully(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        ft._init_forward_test_tables(self.db)
        
        # Create campaign with unknown timeframe
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "unknown", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        
        # Should not hang or raise exception
        mgr.sync_jobs_from_db()
        
        # Scheduler should be started (because sync starts it), but no job should be registered
        self.assertTrue(mgr._is_running)
        self.assertEqual(len(mgr._jobs), 0)  # No jobs registered due to unknown timeframe
        mgr.stop()

    def test_sync_jobs_from_db_starts_scheduler_if_not_running(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        ft._init_forward_test_tables(self.db)
        self.assertFalse(mgr._is_running)
        
        mgr.sync_jobs_from_db()
        
        self.assertTrue(mgr._is_running)
        mgr.stop()

    def test_sync_jobs_from_db_registers_active_campaigns(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        ft._init_forward_test_tables(self.db)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.commit()
        
        with patch.object(mgr, "register_campaign_job", return_value=True) as mock_register:
            mgr.sync_jobs_from_db()
        
        mock_register.assert_called_once_with("v1", "BTC", "15m")
        mgr.stop()

    def test_sync_jobs_from_db_cancels_inactive_campaigns(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        ft._init_forward_test_tables(self.db)
        
        # Register a job first
        mock_job = MagicMock()
        mgr._jobs[("v1", "BTC", "15m")] = mock_job
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "qualified"),
            )
            c.commit()
        
        with patch("schedule.cancel_job") as mock_cancel:
            mgr.sync_jobs_from_db()
        
        mock_cancel.assert_called_once_with(mock_job)
        mgr.stop()

    def test_sync_jobs_from_db_cancels_campaigns_not_in_db(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        ft._init_forward_test_tables(self.db)
        
        # Register a job for campaign not in DB
        mock_job = MagicMock()
        mgr._jobs[("v1", "BTC", "15m")] = mock_job
        
        with patch("schedule.cancel_job") as mock_cancel:
            mgr.sync_jobs_from_db()
        
        mock_cancel.assert_called_once_with(mock_job)
        mgr.stop()

    def test_sync_jobs_from_db_cancels_campaigns_with_enough_runs(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        ft._init_forward_test_tables(self.db)
        
        mock_job = MagicMock()
        mgr._jobs[("v1", "BTC", "15m")] = mock_job
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 1, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, created_at) VALUES (?,?,?,?)",
                (1, "2025-01-01T00:00:00Z", "Range", "2025-01-01T00:00:00Z"),
            )
            c.commit()
        
        with patch("schedule.cancel_job") as mock_cancel:
            mgr.sync_jobs_from_db()
        
        mock_cancel.assert_called_once_with(mock_job)
        mgr.stop()

    def test_list_registered_jobs(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mock_job = MagicMock()
        mock_job.next_run = datetime(2025, 1, 1, 12, 0, 0)
        mgr._jobs[("v1", "BTC", "15m")] = mock_job
        
        jobs = mgr.list_registered_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["version_id"], "v1")
        self.assertEqual(jobs[0]["symbol"], "BTC")
        self.assertEqual(jobs[0]["timeframe"], "15m")
        self.assertEqual(jobs[0]["interval_minutes"], 15)

    def test_list_registered_jobs_no_next_run_attribute(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mock_job = MagicMock()
        del mock_job.next_run  # Remove next_run attribute
        mgr._jobs[("v1", "BTC", "15m")] = mock_job
        
        jobs = mgr.list_registered_jobs()
        self.assertEqual(jobs[0]["next_run"], "unknown")

    def test_trigger_all_pending_calls_trigger_function(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        
        with patch("forward_testing.trigger_all_pending_forward_tests") as mock_trigger:
            mock_trigger.return_value = {"total_campaigns": 0}
            result = mgr.trigger_all_pending()
        
        mock_trigger.assert_called_once_with(
            golden_source=mgr.golden_source,
            db_path=mgr._db_path,
            config=mgr.config,
        )

    def test_job_wrapper_cancels_job_when_result_is_none(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        
        with patch("forward_testing.run_one_forward_test", return_value=None) as mock_run:
            with patch.object(mgr, "cancel_campaign_job") as mock_cancel:
                mgr._job_wrapper("v1", "BTC", "15m")
        
        mock_cancel.assert_called_once_with("v1", "BTC", "15m")

    def test_job_wrapper_handles_exception(self):
        mgr = ft.ForwardTestCronManager(self.config)
        mgr._db_path = self.db
        
        with patch("forward_testing.run_one_forward_test", side_effect=Exception("Test error")):
            # Should not raise
            mgr._job_wrapper("v1", "BTC", "15m")


class TestGetCampaignAccuracyEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        ft._init_forward_test_tables(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_campaign_accuracy_campaign_row_none_handles_gracefully(self):
        # This tests the case where campaign_row might be None (shouldn't happen but test it)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:00:00Z", "Range", "Range", 1, "2025-01-01T00:00:00Z"),
            )
            c.commit()
        
        # Manually delete the campaign row to simulate edge case
        with sqlite3.connect(self.db) as c:
            c.execute("DELETE FROM forward_test_campaigns WHERE id=1")
            c.commit()
        
        result = ft.get_campaign_accuracy("v1", "BTC", "15m", db_path=self.db)
        self.assertIsNone(result)


class TestAutoPromoteEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "ft.db")
        self.models_dir = os.path.join(self.tmp, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        ft._init_forward_test_tables(self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_auto_promote_models_dir_fallback(self):
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(version_dir, exist_ok=True)
        
        config_no_models_dir = MagicMock()
        # Don't set MODELS_DIR attribute
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # 4 matches out of 5 = 80% accuracy
            for i in range(4):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:04:00Z", "Range", "Trend", 0, "2025-01-01T00:04:00Z"),
            )
            c.commit()
        
        with patch("config.TrainingConfig") as mock_tc:
            mock_tc.MODELS_DIR = None
            with patch("model_registry.set_prod", return_value=True) as mock_set_prod:
                with patch("os.path.dirname", return_value=self.tmp):
                    result = ft._auto_promote_to_prod_if_qualified(
                        "v1", "BTC", "15m", 1,
                        db_path=self.db, config=config_no_models_dir
                    )
        
        # Should use fallback path
        self.assertTrue(result)
        mock_set_prod.assert_called_once()

    def test_auto_promote_config_none_uses_training_config(self):
        version_dir = os.path.join(self.models_dir, "v1", "BTC", "15m")
        os.makedirs(self.models_dir, exist_ok=True)
        
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO forward_test_campaigns (version_id, symbol, timeframe, created_at, required_runs, status) VALUES (?,?,?,?,?,?)",
                ("v1", "BTC", "15m", "2025-01-01T00:00:00Z", 5, "active"),
            )
            # 4 matches out of 5 = 80% accuracy
            for i in range(4):
                c.execute(
                    "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                    (1, f"2025-01-01T00:0{i}:00Z", "Range", "Range", 1, f"2025-01-01T00:0{i}:00Z"),
                )
            c.execute(
                "INSERT INTO forward_test_runs (campaign_id, run_at, model_regime, golden_regime, match, created_at) VALUES (?,?,?,?,?,?)",
                (1, "2025-01-01T00:04:00Z", "Range", "Trend", 0, "2025-01-01T00:04:00Z"),
            )
            c.commit()
        
        mock_tc = MagicMock()
        mock_tc.MODELS_DIR = self.models_dir
        
        with patch("config.TrainingConfig", mock_tc):
            with patch("model_registry.set_prod", return_value=True) as mock_set_prod:
                result = ft._auto_promote_to_prod_if_qualified(
                    "v1", "BTC", "15m", 1,
                    db_path=self.db, config=None
                )
        
        self.assertTrue(result)
        mock_set_prod.assert_called_once()


if __name__ == "__main__":
    unittest.main()
