"""
Módulo 8.2 — Normalização e limpeza de texto.
Ver roadmap.md para especificação completa.

TODO: implementar funções completas
"""
import re
import unicodedata


def normalize_numero(text: str) -> str:
    """'REN 1.000/2021' → 'REN 1000/2021'"""
    # TODO: implementar
    return re.sub(r'\.(?=\d)', '', text)


def normalize_tipo(raw: str) -> str:
    """'Resolução Normativa' → 'REN'"""
    # TODO: implementar mapeamento completo
    mapa = {
        "resolução normativa": "REN",
        "resolução homologatória": "REH",
        "resolução autorizativa": "REA",
        "despacho": "DSP",
        "portaria": "PRT",
    }
    return mapa.get(raw.strip().lower(), raw.upper()[:3])


def clean_whitespace(text: str) -> str:
    """Remove espaços e quebras de linha redundantes."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_control_chars(text: str) -> str:
    """Remove caracteres de controle exceto \\n e \\t."""
    return re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)


def normalize_encoding(text: str) -> str:
    """Corrige caracteres mal-encodados e normaliza para UTF-8 NFC."""
    return unicodedata.normalize('NFC', text)
