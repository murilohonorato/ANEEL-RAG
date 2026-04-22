"""
Módulo 4.10 — Testes de chunking document-aware.
Rode com: pytest tests/test_chunk.py -v
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── carregamento do módulo ────────────────────────────────────────────────────

def _load_chunk() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "chunk",
        Path(__file__).parent.parent / "src" / "04_chunk.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_chunk()

build_ementa_chunk       = _mod.build_ementa_chunk
split_by_articles        = _mod.split_by_articles
build_parent_and_children = _mod.build_parent_and_children
build_fallback_chunks    = _mod.build_fallback_chunks
chunk_document           = _mod.chunk_document
generate_chunk_id        = _mod.generate_chunk_id
build_prefix             = _mod.build_prefix


# ── fixture: linha de metadados sintética ────────────────────────────────────

def _make_row(
    doc_id:    str   = "ren20211000ti",
    tipo_sigla: str  = "REN",
    tipo_nome: str   = "Resolução Normativa",
    numero:    float = 1000.0,
    ano:       int   = 2021,
    ementa:    str   = "Estabelece os Procedimentos de Distribuição de Energia Elétrica.",
    situacao:  str   = "NÃO CONSTA REVOGAÇÃO EXPRESSA",
    autor:     str   = "ANEEL",
    assunto:   str   = "Distribuição",
    revogada:  bool  = False,
) -> pd.Series:
    return pd.Series({
        "doc_id":    doc_id,
        "tipo_sigla": tipo_sigla,
        "tipo_nome":  tipo_nome,
        "numero":     numero,
        "ano":        ano,
        "ementa":     ementa,
        "situacao":   situacao,
        "autor":      autor,
        "assunto":    assunto,
        "revogada":   revogada,
        "data_pub":   "2021-12-07",
        "data_ass":   "2021-12-01",
        "pdf_url":    "http://example.com/ren20211000ti.pdf",
    })


TEXTO_REN = (
    "Art. 1o Esta Resolucao estabelece os Procedimentos de Distribuicao.\n\n"
    "§ 1o Para os fins desta Resolucao, considera-se:\n"
    "I - microgeracao distribuida: central geradora de energia eletrica;\n"
    "II - minigeracao distribuida: central geradora com potencia superior.\n\n"
    "Art. 2o A distribuidora deve garantir o acesso ao sistema de distribuicao.\n\n"
    "§ 1o O prazo para conexao e de 30 dias uteis.\n"
    "§ 2o Em casos especiais, o prazo pode ser prorrogado.\n\n"
    "Art. 3o Esta Resolucao entra em vigor na data de sua publicacao."
)

TEXTO_DSP = (
    "DESPACHO DO DIRETOR-GERAL\n\n"
    "Em atencao ao Oficio n. 123/2022, informamos que o pedido foi analisado.\n"
    "O processo sera encaminhado para deliberacao. Brasilia, 10 de marco de 2022."
)


# ── 4.1 — test_ementa_chunk_always_created ───────────────────────────────────

def test_ementa_chunk_always_created():
    """Todo documento gera exatamente 1 chunk de ementa."""
    row = _make_row()

    # com texto
    chunks = chunk_document(row, TEXTO_REN)
    ementas = [c for c in chunks if c["chunk_type"] == "ementa"]
    assert len(ementas) == 1, f"Esperado 1 ementa, gerado {len(ementas)}"

    # sem texto (PDF não encontrado)
    chunks_sem = chunk_document(row, "")
    ementas_sem = [c for c in chunks_sem if c["chunk_type"] == "ementa"]
    assert len(ementas_sem) == 1, "Deve gerar ementa mesmo sem texto do PDF"


def test_ementa_chunk_fields():
    """Chunk de ementa deve ter campos obrigatórios corretos."""
    row = _make_row()
    chunk = build_ementa_chunk(row)

    assert chunk["chunk_type"]  == "ementa"
    assert chunk["is_ementa"]   is True
    assert chunk["parent_id"]   is None
    assert chunk["texto_parent"] is None
    assert chunk["tipo_sigla"]  == "REN"
    assert chunk["ano"]         == 2021
    assert "REN 1000/2021" in chunk["texto"]
    assert chunk["tokens"]      > 0


# ── 4.2 — test_artigo_detection ──────────────────────────────────────────────

def test_artigo_detection():
    """Texto com 3 artigos deve detectar 3 artigos."""
    articles = split_by_articles(TEXTO_REN)
    assert len(articles) == 3, f"Esperado 3 artigos, detectado {len(articles)}"
    assert articles[0]["art_num"] == 1
    assert articles[1]["art_num"] == 2
    assert articles[2]["art_num"] == 3


def test_artigo_detection_empty():
    """Texto sem artigos retorna lista vazia."""
    articles = split_by_articles(TEXTO_DSP)
    assert articles == []


def test_artigo_detection_preserves_text():
    """Texto de cada artigo deve conter o conteúdo original."""
    articles = split_by_articles(TEXTO_REN)
    art1 = articles[0]["art_text"]
    assert "Procedimentos" in art1 or "Resolucao" in art1


# ── 4.3/4.4 — test_parent_child_link ─────────────────────────────────────────

def test_parent_child_link():
    """Children devem apontar para o parent correto e ter texto_parent não vazio."""
    row  = _make_row()
    art  = split_by_articles(TEXTO_REN)[0]  # Art. 1
    parent, children = build_parent_and_children(row, art["art_num"], art["art_text"], seq_start=0)

    assert len(children) > 0, "Deve gerar ao menos 1 child"
    for child in children:
        assert child["parent_id"]    == parent["chunk_id"], "parent_id incorreto"
        assert child["texto_parent"] == parent["texto"],    "texto_parent incorreto"
        assert child["chunk_type"]   == "child"
        assert child["art_num"]      == art["art_num"]


def test_parent_not_in_indexable_chunks():
    """Parent NÃO deve aparecer nos chunks indexáveis do chunk_document."""
    row    = _make_row()
    chunks = chunk_document(row, TEXTO_REN)
    types  = {c["chunk_type"] for c in chunks}
    assert "parent" not in types, "Parent não deve ser indexado"


# ── 4.5 — test_fallback_for_despacho ─────────────────────────────────────────

def test_fallback_for_despacho():
    """Texto sem artigos deve usar fallback com chunk_type='fallback'."""
    row    = _make_row(doc_id="dsp20221000", tipo_sigla="DSP")
    chunks = chunk_document(row, TEXTO_DSP)

    non_ementa = [c for c in chunks if c["chunk_type"] != "ementa"]
    assert len(non_ementa) > 0
    for c in non_ementa:
        assert c["chunk_type"] == "fallback", (
            f"Esperado 'fallback', encontrado '{c['chunk_type']}'"
        )


def test_fallback_self_reference():
    """No fallback, parent_id deve apontar para o próprio chunk_id."""
    row    = _make_row(doc_id="dsp20221000", tipo_sigla="DSP")
    chunks = [c for c in chunk_document(row, TEXTO_DSP) if c["chunk_type"] == "fallback"]
    for c in chunks:
        assert c["parent_id"] == c["chunk_id"]


# ── 4.6 — test_contextual_prefix ─────────────────────────────────────────────

def test_contextual_prefix():
    """Todo chunk deve conter o prefixo '{TIPO} {NUM}/{ANO}'."""
    row    = _make_row()
    chunks = chunk_document(row, TEXTO_REN)
    for c in chunks:
        assert "REN 1000/2021" in c["texto"], (
            f"Prefixo ausente no chunk {c['chunk_id']}: {c['texto'][:80]}"
        )


def test_contextual_prefix_with_art_num():
    """Prefixo de child deve mencionar o artigo."""
    row    = _make_row()
    chunks = [c for c in chunk_document(row, TEXTO_REN) if c["chunk_type"] == "child"]
    assert len(chunks) > 0
    # ao menos um chunk deve mencionar "Art."
    has_art_ref = any("Art." in c["texto"] or "Art" in c["texto"][:200] for c in chunks)
    assert has_art_ref


# ── token limits ─────────────────────────────────────────────────────────────

def test_token_limits_child():
    """Children não devem ultrapassar CHILD_MAX_TOKENS."""
    from src.utils.token_counter import count_tokens
    row    = _make_row()
    chunks = [c for c in chunk_document(row, TEXTO_REN) if c["chunk_type"] == "child"]
    for c in chunks:
        assert c["tokens"] <= _mod.CHILD_MAX_TOKENS + 10, (  # +10 de tolerância para prefixo
            f"Child {c['chunk_id']} tem {c['tokens']} tokens (max {_mod.CHILD_MAX_TOKENS})"
        )


def test_token_limits_parent():
    """Parents não devem ultrapassar PARENT_MAX_TOKENS."""
    row = _make_row()
    art = split_by_articles(TEXTO_REN)[0]
    parent, _ = build_parent_and_children(row, art["art_num"], art["art_text"], seq_start=0)
    assert parent["tokens"] <= _mod.PARENT_MAX_TOKENS, (
        f"Parent tem {parent['tokens']} tokens (max {_mod.PARENT_MAX_TOKENS})"
    )


# ── 4.7 — test_chunk_id_uniqueness ───────────────────────────────────────────

def test_chunk_id_uniqueness():
    """Todos os chunk_ids de um documento devem ser únicos."""
    row    = _make_row()
    chunks = chunk_document(row, TEXTO_REN)
    ids    = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), (
        f"chunk_ids duplicados: {[x for x in ids if ids.count(x) > 1]}"
    )


def test_chunk_id_format():
    """chunk_id deve seguir o padrão {doc_id}_{prefix}{seq:04d}."""
    import re
    row    = _make_row()
    chunks = chunk_document(row, TEXTO_REN)
    for c in chunks:
        assert re.match(r'^.+_[ecft]\d{4}$', c["chunk_id"]), (
            f"Formato inválido: {c['chunk_id']}"
        )


# ── tabelas ───────────────────────────────────────────────────────────────────

TEXTO_TABLE = (
    "Art. 1o Esta norma define tarifas.\n\n"
    "Tabela 1 — Tarifas por classe de consumidor\n"
    "| Classe | Subgrupo | Tarifa (R$/kWh) |\n"
    "|--------|----------|----------------|\n"
    "| Residencial | B1 | 0,72 |\n"
    "| Comercial   | B3 | 0,85 |\n"
    "| Rural        | B2 | 0,58 |\n"
    "| Industrial  | A4 | 0,41 |\n"
    "| Iluminação  | B4b | 0,65 |\n"
    "| Poder Público | B4a | 0,69 |\n\n"
    "Art. 2o Estas tarifas entram em vigor em 01/01/2022."
)

extract_markdown_tables = _mod.extract_markdown_tables
build_table_chunks      = _mod.build_table_chunks


def test_table_extraction_finds_table():
    """Texto com tabela markdown deve detectar 1 tabela."""
    tables = extract_markdown_tables(TEXTO_TABLE)
    assert len(tables) == 1, f"Esperado 1 tabela, detectado {len(tables)}"


def test_table_extraction_header_and_rows():
    """Tabela extraída deve ter header e pelo menos 1 linha de dados."""
    tables = extract_markdown_tables(TEXTO_TABLE)
    tbl = tables[0]
    assert "Classe" in tbl["header"], "Header não contém nome de coluna"
    assert len(tbl["rows"]) >= 1, "Sem linhas de dados"


def test_table_chunk_type():
    """Chunks de tabela devem ter chunk_type='table'."""
    row    = _make_row()
    chunks = chunk_document(row, TEXTO_TABLE)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) > 0, "Nenhum chunk de tabela gerado"


def test_table_chunk_has_header_in_every_chunk():
    """Header (linha descritiva) deve estar presente em TODOS os chunks de tabela."""
    row    = _make_row()
    tables = extract_markdown_tables(TEXTO_TABLE)
    tbl_chunks = build_table_chunks(row, tables, seq_start=0)
    for c in tbl_chunks:
        assert tables[0]["header"] in c["texto"], (
            f"Header ausente no chunk {c['chunk_id']}: {c['texto'][:120]}"
        )


def test_table_chunk_rows_per_chunk():
    """Cada chunk não deve exceder TABLE_ROWS_PER_CHUNK linhas de dados."""
    row    = _make_row()
    tables = extract_markdown_tables(TEXTO_TABLE)
    tbl_chunks = build_table_chunks(row, tables, seq_start=0)
    limit = _mod.TABLE_ROWS_PER_CHUNK
    for c in tbl_chunks:
        # contar linhas que não são o header, separator ou prefix
        header = tables[0]["header"]
        data_lines = [
            ln for ln in c["texto"].splitlines()
            if ln.startswith("|") and ln != header and not re.match(r'^\|[-| :]+\|$', ln)
        ]
        assert len(data_lines) <= limit, (
            f"Chunk {c['chunk_id']} tem {len(data_lines)} linhas (max {limit})"
        )


def test_table_chunk_self_reference():
    """Chunks de tabela: parent_id deve apontar para o próprio chunk_id."""
    row    = _make_row()
    tables = extract_markdown_tables(TEXTO_TABLE)
    tbl_chunks = build_table_chunks(row, tables, seq_start=0)
    for c in tbl_chunks:
        assert c["parent_id"] == c["chunk_id"], (
            f"parent_id '{c['parent_id']}' != chunk_id '{c['chunk_id']}'"
        )


def test_no_table_chunks_without_table():
    """Texto sem tabela não deve gerar chunks do tipo 'table'."""
    row    = _make_row()
    chunks = chunk_document(row, TEXTO_REN)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) == 0, f"Gerou {len(table_chunks)} chunks de tabela sem tabela no texto"
