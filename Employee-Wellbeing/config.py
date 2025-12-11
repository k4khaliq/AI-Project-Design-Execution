# config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Data
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_DATA_VERSION = "v2"
N_EMPLOYEES_DEFAULT = 5000

# Models
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
BURNOUT_MODEL_PATH = MODELS_DIR / "burnout_model.pkl"
FEATURES_METADATA_PATH = MODELS_DIR / "features_metadata.pkl"

# RAG
HR_POLICY_DIR = BASE_DIR / "hr_policies"
RAG_DB_DIR = BASE_DIR / "rag_db"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM config
LLM_PROVIDER = "ollama"
OPENAI_MODEL = "gpt-4.1-mini"  # ignored now
OLLAMA_MODEL = "llama3.1"      # must match the tag you pulled with ollama

# Randomness
GLOBAL_SEED = 42
