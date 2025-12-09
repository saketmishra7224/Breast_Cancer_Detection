import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(BASE_DIR, "artifacts"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
