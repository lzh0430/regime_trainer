"""
Unit tests for model version control (model_registry and config version/PROD paths).
Target: 100% coverage with temp dirs/DB and mocks for edge cases.

Run tests:
  python -m unittest test_model_version_control -v

Check coverage (requires: pip install coverage):
  python -m coverage run --source=model_registry,config -m unittest test_model_version_control
  python -m coverage report -m
"""
import os
import sys
import sqlite3
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_registry as reg


class TestDefaultPaths(unittest.TestCase):
    """Test _default_db_path and _default_models_dir."""

    def test_default_db_path(self):
        out = reg._default_db_path()
        self.assertIn("data", out)
        self.assertTrue(out.endswith("model_registry.db"))

    def test_default_models_dir(self):
        out = reg._default_models_dir()
        self.assertIn("models", out)


class TestGetConnAndInit(unittest.TestCase):
    """Test _get_conn and _init_database."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "sub", "registry.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_conn_creates_parent_dir(self):
        conn = reg._get_conn(self.db)
        conn.close()
        self.assertTrue(os.path.isdir(os.path.dirname(self.db)))

    def test_init_database_creates_tables(self):
        reg._init_database(self.db)
        with sqlite3.connect(self.db) as c:
            r = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            names = {x[0] for x in r}
        self.assertIn("model_versions", names)
        self.assertIn("prod_pointer", names)


class TestAllocateVersionId(unittest.TestCase):
    """Test allocate_version_id with various dir states."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.today = __import__("datetime").datetime.now().strftime("%Y-%m-%d")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_allocate_when_models_dir_does_not_exist(self):
        missing = os.path.join(self.tmp, "nonexistent")
        out = reg.allocate_version_id(models_dir=missing)
        self.assertEqual(out, f"{self.today}-1")

    def test_allocate_when_models_dir_empty(self):
        out = reg.allocate_version_id(models_dir=self.tmp)
        self.assertEqual(out, f"{self.today}-1")

    def test_allocate_increments_from_existing_same_day(self):
        for i in (1, 2, 3):
            d = os.path.join(self.tmp, f"{self.today}-{i}")
            os.makedirs(d)
        out = reg.allocate_version_id(models_dir=self.tmp)
        self.assertEqual(out, f"{self.today}-4")

    def test_allocate_ignores_non_matching_prefix(self):
        os.makedirs(os.path.join(self.tmp, "2020-01-01-1"))
        os.makedirs(os.path.join(self.tmp, "other"))
        out = reg.allocate_version_id(models_dir=self.tmp)
        self.assertEqual(out, f"{self.today}-1")

    def test_allocate_ignores_file_not_dir(self):
        p = os.path.join(self.tmp, f"{self.today}-5")
        Path(p).touch()
        out = reg.allocate_version_id(models_dir=self.tmp)
        self.assertEqual(out, f"{self.today}-1")

    def test_allocate_ignores_non_digit_suffix(self):
        os.makedirs(os.path.join(self.tmp, f"{self.today}-abc"))
        os.makedirs(os.path.join(self.tmp, f"{self.today}-2"))
        out = reg.allocate_version_id(models_dir=self.tmp)
        self.assertEqual(out, f"{self.today}-3")

    def test_allocate_uses_default_models_dir_when_none(self):
        with patch.object(reg, "_default_models_dir", return_value=self.tmp):
            out = reg.allocate_version_id(models_dir=None)
        self.assertEqual(out, f"{self.today}-1")

    def test_allocate_handles_value_error_on_index_parse(self):
        """Cover except ValueError: continue when parsing index (edge case)."""
        os.makedirs(os.path.join(self.tmp, f"{self.today}-1"))
        with patch("builtins.int", side_effect=ValueError("mock")):
            out = reg.allocate_version_id(models_dir=self.tmp)
        # int() raises, we continue; indexes stays empty -> next_index = 1
        self.assertEqual(out, f"{self.today}-1")


class TestRegisterVersion(unittest.TestCase):
    """Test register_version."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "reg.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_register_version_creates_row(self):
        reg.register_version("2025-01-31-1", db_path=self.db)
        with sqlite3.connect(self.db) as c:
            r = c.execute("SELECT version_id, created_at FROM model_versions WHERE version_id=?", ("2025-01-31-1",)).fetchone()
        self.assertIsNotNone(r)
        self.assertIn("Z", r[1])

    def test_register_version_idempotent(self):
        reg.register_version("2025-01-31-2", db_path=self.db)
        reg.register_version("2025-01-31-2", db_path=self.db)
        with sqlite3.connect(self.db) as c:
            n = c.execute("SELECT COUNT(*) FROM model_versions WHERE version_id=?", ("2025-01-31-2",)).fetchone()[0]
        self.assertEqual(n, 1)

    def test_register_uses_default_db_when_none(self):
        with patch.object(reg, "_default_db_path", return_value=self.db):
            reg.register_version("2025-01-31-3", db_path=None)
        self.assertTrue(os.path.isfile(self.db))


class TestSetProd(unittest.TestCase):
    """Test set_prod validation and update."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "reg.db")
        self.models = os.path.join(self.tmp, "models")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_set_prod_fails_when_path_missing(self):
        out = reg.set_prod("BTCUSDT", "15m", "2025-01-31-1", models_dir=self.models, db_path=self.db)
        self.assertFalse(out)

    def test_set_prod_succeeds_when_path_exists(self):
        dir_path = os.path.join(self.models, "2025-01-31-1", "BTCUSDT", "15m")
        os.makedirs(dir_path)
        out = reg.set_prod("BTCUSDT", "15m", "2025-01-31-1", models_dir=self.models, db_path=self.db)
        self.assertTrue(out)
        with sqlite3.connect(self.db) as c:
            r = c.execute("SELECT version_id FROM prod_pointer WHERE symbol=? AND timeframe=?", ("BTCUSDT", "15m")).fetchone()
        self.assertEqual(r[0], "2025-01-31-1")

    def test_set_prod_updates_existing_row(self):
        dir_path = os.path.join(self.models, "2025-01-31-1", "BTCUSDT", "15m")
        os.makedirs(dir_path)
        reg.set_prod("BTCUSDT", "15m", "2025-01-31-1", models_dir=self.models, db_path=self.db)
        os.makedirs(os.path.join(self.models, "2025-01-31-2", "BTCUSDT", "15m"), exist_ok=True)
        reg.set_prod("BTCUSDT", "15m", "2025-01-31-2", models_dir=self.models, db_path=self.db)
        with sqlite3.connect(self.db) as c:
            r = c.execute("SELECT version_id FROM prod_pointer WHERE symbol=? AND timeframe=?", ("BTCUSDT", "15m")).fetchone()
        self.assertEqual(r[0], "2025-01-31-2")


class TestGetProdInfo(unittest.TestCase):
    """Test get_prod_info."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "reg.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_prod_info_no_db_returns_none(self):
        out = reg.get_prod_info("BTCUSDT", "15m", db_path=self.db)
        self.assertIsNone(out)

    def test_get_prod_info_db_exists_no_row_returns_none(self):
        reg._init_database(self.db)
        out = reg.get_prod_info("BTCUSDT", "15m", db_path=self.db)
        self.assertIsNone(out)

    def test_get_prod_info_returns_row(self):
        reg._init_database(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO prod_pointer (symbol, timeframe, version_id, updated_at) VALUES (?,?,?,?)",
                ("BTCUSDT", "15m", "2025-01-31-1", "2025-01-31T12:00:00Z"),
            )
            c.commit()
        out = reg.get_prod_info("BTCUSDT", "15m", db_path=self.db)
        self.assertEqual(out["symbol"], "BTCUSDT")
        self.assertEqual(out["timeframe"], "15m")
        self.assertEqual(out["version_id"], "2025-01-31-1")
        self.assertEqual(out["updated_at"], "2025-01-31T12:00:00Z")


class TestGetProdVersion(unittest.TestCase):
    """Test get_prod_version (DB row vs fallback to get_latest_version)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "reg.db")
        self.models = os.path.join(self.tmp, "models")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_prod_version_no_db_calls_get_latest_version(self):
        out = reg.get_prod_version("BTCUSDT", "15m", models_dir=self.models, db_path=self.db)
        self.assertIsNone(out)

    def test_get_prod_version_with_row_returns_version_id(self):
        reg._init_database(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute(
                "INSERT INTO prod_pointer (symbol, timeframe, version_id, updated_at) VALUES (?,?,?,?)",
                ("BTCUSDT", "15m", "v1", "2025-01-31T12:00:00Z"),
            )
            c.commit()
        out = reg.get_prod_version("BTCUSDT", "15m", db_path=self.db)
        self.assertEqual(out, "v1")

    def test_get_prod_version_no_row_fallback_to_latest(self):
        reg._init_database(self.db)
        os.makedirs(os.path.join(self.models, "2025-01-31-1", "BTCUSDT", "15m"))
        out = reg.get_prod_version("BTCUSDT", "15m", models_dir=self.models, db_path=self.db)
        self.assertEqual(out, "2025-01-31-1")


class TestGetLatestVersion(unittest.TestCase):
    """Test get_latest_version (no dir, no candidates, with DB order, with DB and unknown candidates, no DB sort)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "reg.db")
        self.models = os.path.join(self.tmp, "models")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_latest_version_models_dir_not_dir_returns_none(self):
        out = reg.get_latest_version("BTCUSDT", "15m", models_dir=self.tmp + "_nonexistent")
        self.assertIsNone(out)

    def test_get_latest_version_no_candidates_returns_none(self):
        os.makedirs(self.models)
        out = reg.get_latest_version("BTCUSDT", "15m", models_dir=self.models, db_path=self.db)
        self.assertIsNone(out)

    def test_get_latest_version_ignores_file_in_models_dir(self):
        os.makedirs(self.models)
        Path(os.path.join(self.models, "2025-01-31-1")).touch()
        out = reg.get_latest_version("BTCUSDT", "15m", models_dir=self.models, db_path=self.db)
        self.assertIsNone(out)

    def test_get_latest_version_with_db_orders_by_created_at(self):
        reg._init_database(self.db)
        with sqlite3.connect(self.db) as c:
            c.executemany(
                "INSERT INTO model_versions (version_id, created_at) VALUES (?,?)",
                [("2025-01-31-2", "2025-01-31T10:00:00Z"), ("2025-01-31-1", "2025-01-31T12:00:00Z")],
            )
            c.commit()
        for v in ("2025-01-31-1", "2025-01-31-2"):
            os.makedirs(os.path.join(self.models, v, "BTCUSDT", "15m"))
        out = reg.get_latest_version("BTCUSDT", "15m", models_dir=self.models, db_path=self.db)
        self.assertEqual(out, "2025-01-31-1")

    def test_get_latest_version_candidates_not_in_db_appended(self):
        reg._init_database(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute("INSERT INTO model_versions (version_id, created_at) VALUES (?,?)", ("2025-01-31-2", "2025-01-31T10:00:00Z"))
            c.commit()
        os.makedirs(os.path.join(self.models, "2025-01-31-1", "BTCUSDT", "15m"))
        os.makedirs(os.path.join(self.models, "2025-01-31-2", "BTCUSDT", "15m"))
        out = reg.get_latest_version("BTCUSDT", "15m", models_dir=self.models, db_path=self.db)
        self.assertIn(out, ("2025-01-31-1", "2025-01-31-2"))

    def test_get_latest_version_no_db_sorts_reverse(self):
        os.makedirs(os.path.join(self.models, "2025-01-31-1", "BTCUSDT", "15m"))
        os.makedirs(os.path.join(self.models, "2025-01-31-2", "BTCUSDT", "15m"))
        out = reg.get_latest_version("BTCUSDT", "15m", models_dir=self.models, db_path=self.db + ".nonexistent")
        self.assertEqual(out, "2025-01-31-2")

    def test_get_latest_version_skips_subdir_without_symbol_timeframe(self):
        os.makedirs(os.path.join(self.models, "2025-01-31-1", "ETHUSDT", "15m"))
        out = reg.get_latest_version("BTCUSDT", "15m", models_dir=self.models, db_path=self.db)
        self.assertIsNone(out)  # no BTCUSDT/15m under any version


class TestListVersions(unittest.TestCase):
    """Test list_versions."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "reg.db")
        self.models = os.path.join(self.tmp, "models")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_versions_models_dir_not_dir_returns_empty(self):
        out = reg.list_versions(models_dir=self.tmp + "_nonexistent", db_path=self.db)
        self.assertEqual(out, [])

    def test_list_versions_with_prod_map_and_version_dirs(self):
        reg._init_database(self.db)
        with sqlite3.connect(self.db) as c:
            c.execute("INSERT INTO model_versions (version_id, created_at) VALUES (?,?)", ("v1", "2025-01-31T12:00:00Z"))
            c.execute("INSERT INTO prod_pointer (symbol, timeframe, version_id, updated_at) VALUES (?,?,?,?)", ("BTCUSDT", "15m", "v1", "2025-01-31T12:00:00Z"))
            c.commit()
        os.makedirs(os.path.join(self.models, "v1", "BTCUSDT", "15m"))
        os.makedirs(os.path.join(self.models, "v2", "BTCUSDT", "15m"))
        out = reg.list_versions(models_dir=self.models, db_path=self.db)
        self.assertEqual(len(out), 2)
        v1 = next(x for x in out if x["version_id"] == "v1")
        v2 = next(x for x in out if x["version_id"] == "v2")
        self.assertEqual(v1["created_at"], "2025-01-31T12:00:00Z")
        syms1 = v1["symbols"]
        self.assertEqual(len(syms1), 1)
        self.assertEqual(syms1[0]["symbol"], "BTCUSDT")
        self.assertEqual(syms1[0]["timeframe"], "15m")
        self.assertTrue(syms1[0]["is_prod"])
        syms2 = v2["symbols"]
        self.assertFalse(syms2[0]["is_prod"])

    def test_list_versions_skips_files_in_version_dir(self):
        os.makedirs(os.path.join(self.models, "v1", "BTCUSDT", "15m"))
        Path(os.path.join(self.models, "v1", "file.txt")).touch()
        out = reg.list_versions(models_dir=self.models, db_path=self.db + ".x")
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]["symbols"]), 1)

    def test_list_versions_no_db_still_returns_versions(self):
        os.makedirs(os.path.join(self.models, "v1", "BTCUSDT", "15m"))
        out = reg.list_versions(models_dir=self.models, db_path=self.db + ".nonexistent")
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["created_at"])


# --- Config version/PROD path tests (with mocks) ---

class TestConfigVersionPaths(unittest.TestCase):
    """Test config get_version_dir, get_model_path_for_version, get_scaler_path_for_version, get_hmm_path_for_version."""

    def setUp(self):
        self.config = __import__("config").TrainingConfig

    def test_get_version_dir(self):
        out = self.config.get_version_dir("2025-01-31-1")
        self.assertIn("models", out)
        self.assertTrue(out.endswith("2025-01-31-1") or out.replace("\\", "/").endswith("2025-01-31-1"))

    def test_get_model_path_for_version_lstm(self):
        out = self.config.get_model_path_for_version("v1", "BTCUSDT", "lstm", "15m")
        self.assertIn("v1", out)
        self.assertIn("BTCUSDT", out)
        self.assertIn("15m", out)
        self.assertTrue("lstm_model.h5" in out)

    def test_get_model_path_for_version_hmm(self):
        out = self.config.get_model_path_for_version("v1", "BTCUSDT", "hmm", "15m")
        self.assertTrue("hmm_model.pkl" in out)

    def test_get_model_path_for_version_default_timeframe(self):
        out = self.config.get_model_path_for_version("v1", "BTCUSDT", "lstm", None)
        self.assertIn(self.config.PRIMARY_TIMEFRAME, out)

    def test_get_scaler_path_for_version(self):
        out = self.config.get_scaler_path_for_version("v1", "BTCUSDT", "15m")
        self.assertIn("scaler.pkl", out)
        self.assertIn("v1", out)

    def test_get_hmm_path_for_version(self):
        out = self.config.get_hmm_path_for_version("v1", "BTCUSDT", "15m")
        self.assertIn("hmm_model.pkl", out)
        self.assertIn("v1", out)


class TestConfigProdPaths(unittest.TestCase):
    """Test config get_prod_model_path, get_prod_scaler_path, get_prod_hmm_path with mocked get_prod_version."""

    def setUp(self):
        self.config = __import__("config").TrainingConfig

    def test_get_prod_model_path_with_version_id(self):
        with patch("model_registry.get_prod_version", return_value="2025-01-31-1"):
            out = self.config.get_prod_model_path("BTCUSDT", "lstm", "15m")
        self.assertIn("2025-01-31-1", out)
        self.assertIn("lstm_model.h5", out)

    def test_get_prod_model_path_fallback_legacy(self):
        with patch("model_registry.get_prod_version", return_value=None):
            out = self.config.get_prod_model_path("BTCUSDT", "lstm", "15m")
        self.assertIn("BTCUSDT", out)
        self.assertIn("lstm_model.h5", out)

    def test_get_prod_scaler_path_with_version_id(self):
        with patch("model_registry.get_prod_version", return_value="v1"):
            out = self.config.get_prod_scaler_path("BTCUSDT", "15m")
        self.assertIn("v1", out)
        self.assertIn("scaler.pkl", out)

    def test_get_prod_scaler_path_fallback_legacy(self):
        with patch("model_registry.get_prod_version", return_value=None):
            out = self.config.get_prod_scaler_path("BTCUSDT", "15m")
        self.assertIn("scaler.pkl", out)

    def test_get_prod_hmm_path_with_version_id(self):
        with patch("model_registry.get_prod_version", return_value="v1"):
            out = self.config.get_prod_hmm_path("BTCUSDT", "15m")
        self.assertIn("v1", out)
        self.assertIn("hmm_model.pkl", out)

    def test_get_prod_hmm_path_fallback_legacy(self):
        with patch("model_registry.get_prod_version", return_value=None):
            out = self.config.get_prod_hmm_path("BTCUSDT", "15m")
        self.assertIn("hmm_model.pkl", out)

    def test_get_prod_model_path_default_timeframe(self):
        with patch("model_registry.get_prod_version", return_value="v1"):
            out = self.config.get_prod_model_path("BTCUSDT", "lstm", None)
        self.assertIn(self.config.PRIMARY_TIMEFRAME, out)


if __name__ == "__main__":
    unittest.main()
