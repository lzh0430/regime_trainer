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


if __name__ == "__main__":
    unittest.main()
