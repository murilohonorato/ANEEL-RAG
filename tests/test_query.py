"""
Módulo 6.10 — Testes do pipeline de consulta.
Rode com: pytest tests/test_query.py -v

Testes unitários que NÃO precisam do Qdrant, bge-m3 nem OpenAI.
Cobrem: query processing, RRF, parent lookup e formatação.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── carregamento do módulo ────────────────────────────────────────────────────

def _load_query() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "query",
        Path(__file__).parent.parent / "src" / "06_query.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_query()

process_query          = _mod.process_query
reciprocal_rank_fusion = _mod.reciprocal_rank_fusion
lookup_parents         = _mod.lookup_parents
format_context         = _mod.format_context


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_hit(id_: str, chunk_type: str, doc_id: str, is_ementa: bool = False,
              texto: str = "texto exemplo", texto_parent: str = "parent exemplo",
              parent_id: str = None):
    return {
        "id":        id_,
        "rrf_score": 0.1,
        "payload": {
            "chunk_type":   chunk_type,
            "doc_id":       doc_id,
            "tipo_sigla":   "REN",
            "numero":       1000,
            "ano":          2021,
            "data_pub":     "2021-12-07",
            "situacao":     "NAO CONSTA REVOGACAO EXPRESSA",
            "ementa":       "Estabelece procedimentos.",
            "is_ementa":    is_ementa,
            "texto":        texto,
            "texto_parent": texto_parent,
            "parent_id":    parent_id or id_,
        },
    }


# ── 6.1 — test_filter_extraction ─────────────────────────────────────────────

def test_filter_extraction_ano():
    """Query com ano deve extrair filtro correto."""
    result = process_query("Quais normas foram publicadas em 2021?")
    assert result["filters"]["ano"] == 2021


def test_filter_extraction_tipo():
    """Query com tipo deve extrair filtro correto."""
    result = process_query("O que diz a REN 1000 sobre microgeracao?")
    assert result["filters"]["tipo_sigla"] == "REN"


def test_filter_extraction_numero():
    """Query com número deve extrair filtro correto."""
    result = process_query("Quais os prazos da REN 1000?")
    assert result["filters"]["numero"] == 1000


def test_filter_extraction_none():
    """Query sem filtros retorna None em todos os campos."""
    result = process_query("O que é microgeração distribuída?")
    assert result["filters"]["ano"]        is None
    assert result["filters"]["tipo_sigla"] is None
    assert result["filters"]["numero"]     is None


def test_query_clean_not_empty():
    """Query limpa nunca deve ser vazia."""
    result = process_query("de da do")
    assert result["clean"]  # fallback para original


def test_query_original_preserved():
    """Original da query deve ser preservado intacto."""
    q = "Qual o prazo da REN 1000/2021?"
    assert process_query(q)["original"] == q


# ── 6.4 — test_rrf ───────────────────────────────────────────────────────────

def _make_qdrant_hit(id_: str, payload: dict):
    """Simula objeto de resultado do Qdrant."""
    class FakeHit:
        def __init__(self, id_, payload):
            self.id      = id_
            self.payload = payload
    return FakeHit(id_, payload)


def test_rrf_combines_results():
    """RRF deve combinar resultados de dense e sparse."""
    dense  = [_make_qdrant_hit("a", {"chunk_type": "child",  "doc_id": "doc1"}),
              _make_qdrant_hit("b", {"chunk_type": "child",  "doc_id": "doc2"})]
    sparse = [_make_qdrant_hit("b", {"chunk_type": "child",  "doc_id": "doc2"}),
              _make_qdrant_hit("c", {"chunk_type": "child",  "doc_id": "doc3"})]

    results = reciprocal_rank_fusion(dense, sparse)
    ids = [r["id"] for r in results]
    assert "a" in ids and "b" in ids and "c" in ids


def test_rrf_boost_ementa():
    """Ementa deve receber score maior que child equivalente."""
    dense  = [_make_qdrant_hit("ementa1", {"chunk_type": "ementa", "is_ementa": True,  "doc_id": "doc1"}),
              _make_qdrant_hit("child1",  {"chunk_type": "child",  "is_ementa": False, "doc_id": "doc2"})]
    sparse = [_make_qdrant_hit("child1",  {"chunk_type": "child",  "is_ementa": False, "doc_id": "doc2"}),
              _make_qdrant_hit("ementa1", {"chunk_type": "ementa", "is_ementa": True,  "doc_id": "doc1"})]

    results  = reciprocal_rank_fusion(dense, sparse)
    score_em = next(r["rrf_score"] for r in results if r["id"] == "ementa1")
    score_ch = next(r["rrf_score"] for r in results if r["id"] == "child1")
    assert score_em > score_ch, "Ementa deve ter score maior que child equivalente"


def test_rrf_ordering():
    """Resultados do RRF devem estar ordenados por score decrescente."""
    dense  = [_make_qdrant_hit(str(i), {"chunk_type": "child", "doc_id": f"doc{i}"}) for i in range(5)]
    sparse = [_make_qdrant_hit(str(i), {"chunk_type": "child", "doc_id": f"doc{i}"}) for i in range(5)]
    results = reciprocal_rank_fusion(dense, sparse)
    scores  = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ── 6.5 — test_parent_lookup ─────────────────────────────────────────────────

def test_parent_deduplication():
    """Dois children do mesmo parent devem virar 1 contexto."""
    hits = [
        _make_hit("c1", "child", "doc1", parent_id="doc1_art001", texto_parent="parent texto A"),
        _make_hit("c2", "child", "doc1", parent_id="doc1_art001", texto_parent="parent texto A"),
    ]
    contexts = lookup_parents(hits)
    assert len(contexts) == 1, f"Esperado 1 contexto, gerado {len(contexts)}"


def test_child_uses_texto_parent():
    """Child deve usar texto_parent, não texto do próprio chunk."""
    hits = [_make_hit("c1", "child", "doc1",
                      texto="texto do child",
                      texto_parent="texto completo do artigo")]
    ctx = lookup_parents(hits)[0]
    assert ctx["texto_parent"] == "texto completo do artigo"


def test_ementa_uses_own_text():
    """Ementa deve usar seu próprio texto como contexto."""
    hits = [_make_hit("e1", "ementa", "doc1",
                      texto="texto da ementa",
                      texto_parent="")]
    ctx = lookup_parents(hits)[0]
    assert "ementa" in ctx["texto_parent"] or ctx["texto_parent"] == "texto da ementa"


def test_lookup_max_20_contexts():
    """Parent lookup retorna no máximo 20 contextos únicos."""
    hits = [_make_hit(f"c{i}", "child", f"doc{i}", parent_id=f"doc{i}_art001")
            for i in range(30)]
    contexts = lookup_parents(hits)
    assert len(contexts) <= 20


# ── 6.7 — test_format_context ────────────────────────────────────────────────

def test_format_context_has_fonte_label():
    """Contexto formatado deve conter '[FONTE N]'."""
    ctx = [_make_hit("e1", "ementa", "doc1", texto_parent="texto exemplo")]
    contexts = lookup_parents(ctx)
    formatted = format_context(contexts)
    assert "[FONTE 1]" in formatted


def test_format_context_respects_token_limit():
    """Contexto formatado não deve exceder CONTEXT_MAX_TOKENS."""
    from src.utils.token_counter import count_tokens as ct
    # criar contexto enorme
    big_text = "palavra " * 5000
    hits = [_make_hit(f"c{i}", "child", f"doc{i}",
                      texto_parent=big_text, parent_id=f"p{i}")
            for i in range(10)]
    contexts  = lookup_parents(hits)
    formatted = format_context(contexts)
    assert ct(formatted) <= _mod.CONTEXT_MAX_TOKENS + 10  # tolerância de 10 tokens


def test_format_context_multiple_sources():
    """Múltiplas fontes devem aparecer numeradas sequencialmente."""
    hits = [
        _make_hit("c1", "child", "doc1", parent_id="p1", texto_parent="contexto A"),
        _make_hit("c2", "child", "doc2", parent_id="p2", texto_parent="contexto B"),
    ]
    contexts  = lookup_parents(hits)
    formatted = format_context(contexts)
    assert "[FONTE 1]" in formatted
    assert "[FONTE 2]" in formatted
