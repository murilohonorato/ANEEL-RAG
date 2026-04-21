"""
Módulo 0.5 — Testes de sanidade do ambiente.
Rode com: pytest tests/test_environment.py -v
"""
import sys


def test_python_version():
    assert sys.version_info >= (3, 11), "Requer Python 3.11+"


def test_critical_imports():
    import fitz          # PyMuPDF
    import qdrant_client
    import tiktoken
    import pandas
    import pyarrow


def test_qdrant_embedded_opens(tmp_path):
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(tmp_path / "test_db"))
    assert client is not None


def test_device_detection():
    from src.utils.env_check import detect_device
    device = detect_device()
    assert device in ("mps", "cuda", "cpu")


def test_config_loads():
    from src.config import ROOT_DIR, DATA_DIR, EMBEDDING_MODEL, COLLECTION_NAME
    assert ROOT_DIR.exists()
    assert EMBEDDING_MODEL == "BAAI/bge-m3"
    assert COLLECTION_NAME == "aneel_chunks"


def test_logger_setup():
    from src.utils.logger import logger
    assert logger is not None
    logger.debug("test_logger_setup OK")
