"""
02b_rename_pdfs.py
------------------
Renomeia os PDFs de nomes genéricos (pdf_1.pdf, pdf_2.pdf, ...)
para nomes canônicos (dsp20214137ti.pdf) usando os index_{ano}.json.

Os PDFs ficam em:
  data/pdfs/{ano}/{doc_id}.pdf   ← destino canônico

Uso:
    python src/02b_rename_pdfs.py
    python src/02b_rename_pdfs.py --dry-run   # só mostra o que seria feito
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger

# CONFIG
PDFS_BASE = Path("data/pdfs")
ANOS = [2016, 2021, 2022]


def rename_year(ano: int, dry_run: bool = False) -> dict:
    """
    Renomeia PDFs de um ano usando o index_{ano}.json.
    Retorna estatísticas: {ok, skipped, missing, errors}.
    """
    idx_path = PDFS_BASE / str(ano) / f"index_{ano}.json"
    src_dir  = PDFS_BASE / str(ano) / f"pdfs_{ano}"
    dst_dir  = PDFS_BASE / str(ano)

    if not idx_path.exists():
        logger.error(f"[{ano}] index não encontrado: {idx_path}")
        return {"ok": 0, "skipped": 0, "missing": 0, "errors": 0}

    if not src_dir.exists():
        logger.error(f"[{ano}] pasta de origem não encontrada: {src_dir}")
        return {"ok": 0, "skipped": 0, "missing": 0, "errors": 0}

    with open(idx_path, encoding="utf-8") as f:
        index = json.load(f)

    stats = {"ok": 0, "skipped": 0, "missing": 0, "errors": 0}

    for entry in index:
        generic_name   = entry.get("filename", "")       # pdf_1.pdf
        canonical_name = entry.get("arquivo", "")        # dsp20214137ti.pdf

        if not generic_name or not canonical_name:
            logger.warning(f"[{ano}] Entrada inválida: {entry}")
            stats["errors"] += 1
            continue

        # garantir extensão .pdf no nome canônico
        if not canonical_name.lower().endswith(".pdf"):
            stats["skipped"] += 1
            continue

        src = src_dir / generic_name
        dst = dst_dir / canonical_name.lower()

        # já existe no destino
        if dst.exists():
            stats["skipped"] += 1
            continue

        # origem não encontrada
        if not src.exists():
            logger.debug(f"[{ano}] Origem não encontrada: {src.name}")
            stats["missing"] += 1
            continue

        if dry_run:
            logger.info(f"[DRY-RUN] {src.name} → {dst.name}")
            stats["ok"] += 1
        else:
            try:
                shutil.copy2(str(src), str(dst))
                stats["ok"] += 1
            except Exception as exc:
                logger.error(f"[{ano}] Erro ao copiar {src.name}: {exc}")
                stats["errors"] += 1

    return stats


def run(dry_run: bool = False) -> None:
    label = "[DRY-RUN] " if dry_run else ""
    logger.info(f"{label}Iniciando renomeação de PDFs...")

    total_ok      = 0
    total_skipped = 0
    total_missing = 0
    total_errors  = 0

    for ano in ANOS:
        logger.info(f"{label}Processando {ano}...")
        stats = rename_year(ano, dry_run=dry_run)
        logger.info(
            f"[{ano}] ok={stats['ok']}, "
            f"já existia={stats['skipped']}, "
            f"origem ausente={stats['missing']}, "
            f"erros={stats['errors']}"
        )
        total_ok      += stats["ok"]
        total_skipped += stats["skipped"]
        total_missing += stats["missing"]
        total_errors  += stats["errors"]

    logger.info(
        f"{label}Concluído — "
        f"copiados={total_ok}, "
        f"já existiam={total_skipped}, "
        f"ausentes={total_missing}, "
        f"erros={total_errors}"
    )

    # verificação final
    if not dry_run:
        for ano in ANOS:
            dst_dir = PDFS_BASE / str(ano)
            n = len(list(dst_dir.glob("*.pdf")))
            logger.info(f"[{ano}] PDFs canônicos em data/pdfs/{ano}/: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renomeia PDFs para nomes canônicos")
    parser.add_argument("--dry-run", action="store_true", help="Apenas simula, sem copiar")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
