"""
Módulo 8.5 — Verificação de ambiente e hardware.
Ver roadmap.md para especificação completa.

Uso:
    python src/utils/env_check.py
"""
import sys
import os


def check_python_version() -> bool:
    return sys.version_info >= (3, 11)


def detect_device() -> str:
    """Retorna 'mps', 'cuda' ou 'cpu'."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def check_env_vars() -> dict:
    """Verifica variáveis de ambiente obrigatórias."""
    required = ["ANTHROPIC_API_KEY"]
    return {k: bool(os.getenv(k)) for k in required}


def check_imports() -> dict:
    """Testa imports críticos."""
    libs = ["fitz", "qdrant_client", "FlagEmbedding", "anthropic", "tiktoken", "ragas", "pandas", "pyarrow"]
    results = {}
    for lib in libs:
        try:
            __import__(lib)
            results[lib] = True
        except ImportError:
            results[lib] = False
    return results


if __name__ == "__main__":
    print(f"Python >= 3.11: {check_python_version()} ({sys.version})")
    print(f"Device:         {detect_device()}")
    print(f"Env vars:       {check_env_vars()}")
    print(f"Imports:")
    for lib, ok in check_imports().items():
        status = "✓" if ok else "✗ FALTANDO"
        print(f"  {lib}: {status}")
