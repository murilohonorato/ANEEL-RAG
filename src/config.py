"""
Configuração global do projeto — carregada por todos os módulos.

Crie um arquivo .env na raiz com as chaves listadas no .env.example.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Raiz do projeto ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

# ── Dados ──────────────────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PDFS_DIR       = DATA_DIR / "pdfs"
PROCESSED_DIR  = DATA_DIR / "processed"

# ── Outputs ────────────────────────────────────────────────────────────────────
QDRANT_PATH    = ROOT_DIR / os.getenv("QDRANT_PATH", "qdrant_db")
RESULTS_DIR    = ROOT_DIR / "results"
LOGS_DIR       = ROOT_DIR / "logs"

# ── Modelos ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
LLM_MODEL       = "claude-sonnet-4-5"

# ── Chunking ───────────────────────────────────────────────────────────────────
PARENT_MAX_TOKENS  = 800
PARENT_MIN_TOKENS  = 400
CHILD_MAX_TOKENS   = 200
CHILD_MIN_TOKENS   = 100
FALLBACK_CHUNK_SIZE    = 500
FALLBACK_CHUNK_OVERLAP = 80
EMENTA_PREFIX_CHARS    = 150

# ── Qdrant ─────────────────────────────────────────────────────────────────────
COLLECTION_NAME    = "aneel_chunks"
DENSE_DIM          = 1024
HNSW_M             = 16
HNSW_EF_CONSTRUCT  = 200

# ── Retrieval ──────────────────────────────────────────────────────────────────
HYBRID_TOP_K       = 40
RRF_K              = 60
EMENTA_BOOST       = 1.3
RERANK_TOP_N       = 5
CONTEXT_MAX_TOKENS = 8000

# ── API ────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

# ── Arquivos JSON brutos (nomes exatos) ────────────────────────────────────────
JSON_FILES = {
    2016: RAW_DIR / "biblioteca_aneel_gov_br_legislacao_2016_metadados.json",
    2021: RAW_DIR / "biblioteca_aneel_gov_br_legislacao_2021_metadados.json",
    2022: RAW_DIR / "biblioteca_aneel_gov_br_legislacao_2022_metadados.json",
}
