"""
Módulo 8.4 — Geração de IDs canônicos.
Ver roadmap.md para especificação completa.

TODO: implementar
"""
import re
import hashlib


def doc_id_from_filename(filename: str) -> str:
    """
    Extrai doc_id canônico do nome do arquivo PDF.
    Ex: 'ren20211000ti.pdf' → 'ren20211000ti'
         'REH2022001.pdf'   → 'reh2022001'
    """
    name = filename.lower().replace('.pdf', '').strip()
    return name


def chunk_id_hash(doc_id: str, chunk_type: str, seq: int) -> str:
    """
    Gera chunk_id determinístico.
    Ex: chunk_id_hash('ren20211000', 'child', 1) → 'ren20211000_c0001'
    """
    prefix = chunk_type[0].lower()  # 'c', 'e', 'f', 'p'
    return f"{doc_id}_{prefix}{seq:04d}"


def int_id_from_str(s: str) -> int:
    """Converte string para int via MD5 (para IDs do Qdrant)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)
