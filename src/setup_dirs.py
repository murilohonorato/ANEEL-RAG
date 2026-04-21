"""
Módulo 0.3 — Criação da estrutura de pastas do projeto.

Uso:
    python src/setup_dirs.py
"""
from pathlib import Path

# CONFIG
ROOT = Path(__file__).parent.parent

DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "pdfs" / "2016",
    ROOT / "data" / "pdfs" / "2021",
    ROOT / "data" / "pdfs" / "2022",
    ROOT / "data" / "processed",
    ROOT / "qdrant_db",
    ROOT / "results",
    ROOT / "logs",
]

GITKEEP_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "results",
]


def setup_dirs() -> None:
    from src.utils.logger import logger

    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Pasta garantida: {d.relative_to(ROOT)}")

    for d in GITKEEP_DIRS:
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    logger.info("Estrutura de pastas criada com sucesso.")


if __name__ == "__main__":
    setup_dirs()
