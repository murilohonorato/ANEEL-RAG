"""
Módulo 3.8 — Testes de parsing de PDFs.
Rode com: pytest tests/test_parse.py -v

Os testes são auto-contidos: criam PDFs sintéticos em memória
com PyMuPDF, sem depender de arquivos reais do dataset.
"""

import importlib.util
import re
import sys
import unicodedata
from pathlib import Path
from types import ModuleType

import fitz  # PyMuPDF
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── carregamento do módulo (nome começa com dígito → importlib) ───────────────

def _load_parse() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "parse",
        Path(__file__).parent.parent / "src" / "03_parse.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_parse()

detect_pdf_type            = _mod.detect_pdf_type
extract_text_native        = _mod.extract_text_native
detect_legal_structure     = _mod.detect_legal_structure
table_to_markdown          = _mod.table_to_markdown
remove_control_chars       = _mod.remove_control_chars
normalize_encoding         = _mod.normalize_encoding


# ── fixtures: PDFs sintéticos ─────────────────────────────────────────────────

def _make_pdf_with_text(tmp_path: Path, text: str, filename: str = "test.pdf") -> Path:
    """Cria um PDF mínimo com texto nativo usando PyMuPDF."""
    pdf_path = tmp_path / filename
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _make_blank_pdf(tmp_path: Path, n_pages: int = 2) -> Path:
    """Cria um PDF com páginas sem texto (simula escaneado)."""
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ── 3.2 — test_native_pdf_non_empty ──────────────────────────────────────────

def test_native_pdf_non_empty(tmp_path):
    """PDF com texto nativo deve retornar texto não vazio após extração."""
    texto = "Art. 1o Esta resolucao estabelece os criterios de geracao distribuida."
    pdf_path = _make_pdf_with_text(tmp_path, texto)

    pages = extract_text_native(pdf_path)

    assert len(pages) > 0, "Nenhuma página extraída"
    full = " ".join(p["text"] for p in pages)
    assert len(full.strip()) > 0, "Texto extraído está vazio"
    assert pages[0]["char_count"] > 0


# ── 3.1 — test_ocr_pdf_detected ──────────────────────────────────────────────

def test_ocr_pdf_detected(tmp_path):
    """PDF sem texto (escaneado simulado) deve ser classificado como 'scanned'."""
    pdf_path = _make_blank_pdf(tmp_path, n_pages=3)
    pdf_type = detect_pdf_type(pdf_path)
    assert pdf_type == "scanned", (
        f"PDF sem texto deveria ser 'scanned', mas foi classificado como '{pdf_type}'"
    )


def test_native_pdf_detected(tmp_path):
    """PDF com texto denso deve ser classificado como 'native'."""
    texto = "Resolucao Normativa ANEEL. " + ("Texto legal com conteudo valido. " * 15)
    pdf_path = _make_pdf_with_text(tmp_path, texto)
    pdf_type = detect_pdf_type(pdf_path)
    assert pdf_type == "native", (
        f"PDF com texto deveria ser 'native', mas foi '{pdf_type}'"
    )


# ── 3.4 — test_table_markdown_format ─────────────────────────────────────────

def test_table_markdown_format():
    """Tabela deve ser convertida para Markdown com pipes e linha separadora."""
    table = [
        ["Distribuidora", "Tarifa (R$)", "Vigencia"],
        ["CEMIG",         "0,42",        "01/2022"],
        ["COPEL",         "0,38",        "03/2022"],
    ]
    md = table_to_markdown(table)

    assert "|" in md, "Markdown deve conter pipe |"
    assert "Distribuidora" in md
    assert "Tarifa (R$)" in md
    lines = md.splitlines()
    assert len(lines) >= 3, "Deve ter ao menos header + separador + 1 linha de dados"
    sep_line = lines[1]
    assert re.match(r'^\|[\s\-|]+\|$', sep_line), (
        f"Segunda linha deve ser separador Markdown, mas foi: '{sep_line}'"
    )
    assert "CEMIG" in md
    assert "COPEL" in md


def test_table_markdown_none_cells():
    """Células None devem ser tratadas como string vazia."""
    table = [
        ["Col A", "Col B"],
        [None,    "valor"],
        ["outro", None],
    ]
    md = table_to_markdown(table)
    assert "|" in md
    assert "valor" in md
    assert "outro" in md


def test_table_markdown_empty_returns_empty():
    """Tabela vazia deve retornar string vazia."""
    assert table_to_markdown([]) == ""
    assert table_to_markdown([[]]) == ""


# ── 3.6 — test_encoding_clean ────────────────────────────────────────────────

def test_encoding_clean():
    """Caracteres de controle devem ser removidos; conteúdo válido preservado."""
    dirty = "Texto\x00com\x01nulos\x0bverticais e conteudo valido.\x7f"
    clean = remove_control_chars(dirty)

    assert "\x00" not in clean, "\\x00 não deve estar no texto limpo"
    assert "\x01" not in clean
    assert "\x0b" not in clean
    assert "\x7f" not in clean
    assert "conteudo valido" in clean, "Conteúdo válido deve ser preservado"


def test_normalize_encoding_bom():
    """BOM deve ser removido pela normalização de encoding."""
    with_bom = "\ufeffTexto com BOM"
    result = normalize_encoding(with_bom)
    assert result == "Texto com BOM"


def test_normalize_encoding_nfc():
    """Texto normalizado deve estar em forma NFC."""
    # forma NFD: 'a' + combining cedilla + combining tilde
    nfd_text = "microgeraca\u0303o"
    result = normalize_encoding(nfd_text)
    assert unicodedata.is_normalized("NFC", result)


# ── 3.5 — test_legal_structure_ren ───────────────────────────────────────────

def test_legal_structure_ren():
    """Texto de REN com artigos e parágrafos deve ser detectado corretamente."""
    texto_ren = (
        "RESOLUCAO NORMATIVA No 1000, DE 7 DE DEZEMBRO DE 2021\n\n"
        "Art. 1o Esta Resolucao estabelece os Procedimentos de Distribuicao.\n\n"
        "§ 1o Para os fins desta Resolucao, considera-se:\n"
        "I - microgeracao distribuida: central geradora;\n"
        "II - minigeracao distribuida: central geradora maior.\n\n"
        "Art. 2o A distribuidora deve garantir acesso ao sistema.\n\n"
        "§ 1o O prazo para conexao e de 30 dias.\n\n"
        "Art. 3o Esta Resolucao entra em vigor na data de publicacao.\n"
    )
    structure = detect_legal_structure(texto_ren)

    assert structure["has_articles"] is True, "Deve detectar artigos"
    assert structure["article_count"] >= 3, (
        f"Esperado >= 3 artigos, detectado {structure['article_count']}"
    )
    assert structure["has_paragraphs"] is True, "Deve detectar parágrafos (§)"


def test_legal_structure_despacho():
    """Despacho sem artigos deve retornar has_articles=False."""
    texto_dsp = (
        "DESPACHO DO DIRETOR-GERAL\n\n"
        "Em atencao ao Oficio n. 123/2022, informamos que o pedido foi analisado.\n"
        "O processo sera encaminhado para deliberacao na proxima reuniao.\n"
        "Brasilia, 10 de marco de 2022.\n"
    )
    structure = detect_legal_structure(texto_dsp)
    assert structure["has_articles"] is False
    assert structure["article_count"] == 0


def test_legal_structure_with_annexes():
    """Texto com ANEXO deve ter has_annexes=True."""
    texto = "Art. 1o Conforme disposto.\nANEXO\nTabela de tarifas."
    structure = detect_legal_structure(texto)
    assert structure["has_annexes"] is True
