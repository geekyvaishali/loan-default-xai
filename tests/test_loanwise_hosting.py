"""Guards for hosting the public demo (Streamlit Community Cloud)."""
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.loanwise import demo
from src.loanwise.db import Database, open_database
from tests.helpers import predictor_and_dir

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ONLY = {"xgboost", "lightgbm", "shap", "lime", "matplotlib", "seaborn", "pyarrow", "pytest"}

RUNTIME_PROBE = r'''
import sys, importlib.abc, tempfile
BLOCK = {"shap", "lime", "matplotlib", "seaborn", "pyarrow", "xgboost", "lightgbm", "pytest"}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCK:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
sys.meta_path.insert(0, Blocker())
from pathlib import Path
from src.predict import LoanRiskPredictor
from src.loanwise import demo
from src.loanwise.db import Database
from src.loanwise.auth import AuthService
from src.loanwise.applications import ApplicationService
from src.loanwise.forms import DEFAULTS
predictor = LoanRiskPredictor.load()
db = Database(Path(tempfile.mkdtemp()) / "probe.db")
assert demo.ensure_seeded(db, predictor) is True
auth = AuthService(db)
token = auth.login(demo.DEMO_CUSTOMERS[0], demo.DEMO_PASSWORD, "customer")
result = ApplicationService(db, auth, predictor).submit(token, None, DEFAULTS)
assert 0 <= result["probability"] <= 1 and result["factors"]
imported = sorted(m for m in sys.modules if m.split(".")[0] in BLOCK)
assert not imported, imported
print("RUNTIME_OK")
'''


class HostingTests(unittest.TestCase):
    def test_ensure_seeded_seeds_once_and_is_idempotent(self):
        predictor, _ = predictor_and_dir()
        db = Database(Path(tempfile.mkdtemp()) / "seed_once.db")
        self.assertTrue(demo.ensure_seeded(db, predictor))
        with db.connect() as conn:
            users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        self.assertFalse(demo.ensure_seeded(db, predictor))
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0], users)

    def test_open_database_falls_back_when_location_is_unusable(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / "a_file_not_a_directory"
            blocker.write_text("x")
            db = open_database(blocker / "sub" / "loanwise.db")      # parent is a file: cannot be created
            self.assertNotIn("a_file_not_a_directory", db.path)
            with db.connect() as conn:
                self.assertEqual(conn.execute("SELECT 1").fetchone()[0], 1)

    def test_runtime_requirements_are_slim_and_pinned(self):
        pkgs = [line.split("#")[0].strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
        pkgs = [p for p in pkgs if p]
        names = {re.split(r"[=<>!~]", p)[0].lower() for p in pkgs}
        self.assertFalse(names & TRAINING_ONLY, "training-only packages must not be in the hosted runtime requirements")
        self.assertTrue({"streamlit", "pandas", "numpy", "scikit-learn", "joblib", "plotly"} <= names)
        for p in pkgs:
            self.assertIn("==", p, f"{p} should be pinned to a tested version")

    def test_runtime_works_without_training_only_libraries(self):
        if not (ROOT / "artifacts" / "model.joblib").exists():
            self.skipTest("no trained artifacts in this checkout")
        run = subprocess.run([sys.executable, "-W", "ignore", "-c", RUNTIME_PROBE], cwd=ROOT,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        self.assertIn("RUNTIME_OK", run.stdout)


if __name__ == "__main__":
    unittest.main()
