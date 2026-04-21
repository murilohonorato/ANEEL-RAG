"""
Módulo 0.5 — Testes de sanidade do ambiente.
Rode com: pytest tests/test_environment.py -v
"""
# TODO: implementar — ver roadmap.md Módulo 0.5


def test_python_version():
    import sys
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
