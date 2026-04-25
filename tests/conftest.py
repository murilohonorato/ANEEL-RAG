import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_index: testes que precisam do índice Qdrant real (qdrant_db/ populado)",
    )
