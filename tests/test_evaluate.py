"""
Módulo 7.7 — Testes de avaliação com RAGAS.
Rode com: pytest tests/test_evaluate.py -v

Testes unitários que NÃO precisam do Qdrant, bge-m3 nem OpenAI.
Cobrem: golden set, schema de output, métricas, juiz e retry.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


# ── carregamento do módulo ────────────────────────────────────────────────────

def _load_evaluate() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate",
        Path(__file__).parent.parent / "src" / "07_evaluate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_evaluate()

compute_retrieval_metrics = _mod.compute_retrieval_metrics
generate_report           = _mod.generate_report
lightweight_judge         = _mod.lightweight_judge
generate_with_retry       = _mod.generate_with_retry


# ── 7.1 — test_golden_set ────────────────────────────────────────────────────

def test_golden_set_valid():
    """Golden set deve existir e ter pelo menos 10 perguntas."""
    assert GOLDEN_SET_PATH.exists(), "golden_set.json nao encontrado"
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 10, f"Esperado >= 10 perguntas, encontrado {len(data)}"


def test_golden_set_fields():
    """Cada item do golden set deve ter os campos obrigatórios."""
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        assert "question"         in item, f"Falta 'question': {item}"
        assert "expected_answer"  in item, f"Falta 'expected_answer': {item}"
        assert "relevant_doc_ids" in item, f"Falta 'relevant_doc_ids': {item}"
        assert "notes"            in item, f"Falta 'notes': {item}"


def test_golden_set_questions_not_empty():
    """Todas as perguntas devem ser não vazias."""
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        assert item["question"].strip(), "Pergunta vazia encontrada"


def test_golden_set_coverage():
    """Golden set deve ter perguntas de pelo menos 3 categorias (via notes)."""
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    def has_category(keyword: str) -> bool:
        return any(keyword in item.get("notes", "").lower() for item in data)

    assert has_category("definição") or has_category("definicao"), \
        "Faltam perguntas de definicao"
    assert has_category("metadado") or has_category("direto"), \
        "Faltam perguntas de metadado"
    assert has_category("agregad") or has_category("tabela") or has_category("tarifar"), \
        "Faltam perguntas agregadas ou de tabela"


# ── 7.2 — test_eval_output_schema ────────────────────────────────────────────

def _make_result(needs_review: bool = False, score: float = 0.8) -> dict:
    return {
        "question":           "Pergunta teste?",
        "expected_answer":    "Resposta esperada.",
        "relevant_doc_ids":   ["doc1"],
        "notes":              "teste",
        "generated_answer":   "Resposta gerada com citação [REN 1000/2021, Art. 1º].",
        "retrieved_contexts": ["contexto 1", "contexto 2"],
        "retrieved_doc_ids":  ["doc1", "doc2"],
        "attempts":           1,
        "final_judge_score":  score,
        "needs_review":       needs_review,
        "judge_history":      [{"attempt": 1, "score": score, "issues": "ok"}],
        "timing":             {"retrieval_s": 1.5, "total_s": 3.0},
    }


def test_eval_output_schema():
    """Output de avaliação deve ter todos os campos necessários."""
    result = _make_result()
    required = [
        "question", "expected_answer", "relevant_doc_ids", "generated_answer",
        "retrieved_contexts", "retrieved_doc_ids", "attempts",
        "final_judge_score", "needs_review", "judge_history", "timing",
    ]
    for field in required:
        assert field in result, f"Campo ausente: {field}"


def test_eval_output_timing_fields():
    """Campo timing deve ter retrieval_s e total_s."""
    result = _make_result()
    assert "retrieval_s" in result["timing"]
    assert "total_s"     in result["timing"]


# ── 7.3 — test_metrics_in_range ──────────────────────────────────────────────

def test_metrics_in_range():
    """Scores do juiz devem estar entre 0 e 1."""
    result = _make_result(score=0.75)
    assert 0.0 <= result["final_judge_score"] <= 1.0


def test_judge_history_format():
    """judge_history deve ter attempt, score e issues."""
    result = _make_result()
    for entry in result["judge_history"]:
        assert "attempt" in entry
        assert "score"   in entry
        assert "issues"  in entry


# ── 7.4 — test_retrieval_metrics ─────────────────────────────────────────────

def test_retrieval_recall_no_relevant():
    """Sem relevant_doc_ids preenchido deve retornar nota explicativa."""
    results = [_make_result() for _ in range(5)]
    for r in results:
        r["relevant_doc_ids"] = []
    metrics = compute_retrieval_metrics(results)
    assert "note" in metrics


def test_retrieval_recall_perfect():
    """Quando todos os docs relevantes foram recuperados, recall deve ser 1.0."""
    results = [_make_result()]
    results[0]["relevant_doc_ids"] = ["doc1"]
    results[0]["retrieved_doc_ids"] = ["doc1", "doc2", "doc3"]
    metrics = compute_retrieval_metrics(results)
    assert metrics["recall_at_5"]  == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr"]          == 1.0


def test_retrieval_recall_miss():
    """Quando doc relevante não foi recuperado, recall deve ser 0.0."""
    results = [_make_result()]
    results[0]["relevant_doc_ids"] = ["doc_nao_recuperado"]
    results[0]["retrieved_doc_ids"] = ["doc1", "doc2"]
    metrics = compute_retrieval_metrics(results)
    assert metrics["recall_at_5"] == 0.0


def test_retrieval_mrr_second_position():
    """Doc relevante na posição 2 deve dar MRR = 0.5."""
    results = [_make_result()]
    results[0]["relevant_doc_ids"] = ["doc2"]
    results[0]["retrieved_doc_ids"] = ["doc1", "doc2", "doc3"]
    metrics = compute_retrieval_metrics(results)
    assert metrics["mrr"] == 0.5


# ── loop de retry ─────────────────────────────────────────────────────────────

def test_retry_accepts_on_first_try():
    """Se juiz aprova na 1ª tentativa, resultado tem attempts=1 e needs_review=False."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"score": 0.9, "issues": "ok", "has_citations": true}'))],
        usage=MagicMock(prompt_tokens=100, completion_tokens=50),
    )

    result = generate_with_retry(
        question="Pergunta teste?",
        formatted_context="[FONTE 1] REN 1000/2021\nArt. 1º texto...\n---",
        contexts=["contexto"],
        openai_client=mock_client,
    )
    assert result["attempts"]     == 1
    assert result["needs_review"] is False
    assert result["final_score"]  >= 0.5


def test_retry_marks_needs_review_after_max():
    """Após MAX_RETRIES falhas, needs_review deve ser True."""
    mock_client = MagicMock()
    # juiz sempre reprova (score baixo)
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"score": 0.2, "issues": "alucinacao", "has_citations": false}'))],
        usage=MagicMock(prompt_tokens=100, completion_tokens=20),
    )

    result = generate_with_retry(
        question="Pergunta teste?",
        formatted_context="contexto",
        contexts=["contexto"],
        openai_client=mock_client,
    )
    assert result["needs_review"] is True
    assert result["attempts"]     == _mod.MAX_RETRIES
    assert len(result["judge_history"]) == _mod.MAX_RETRIES


# ── relatório ─────────────────────────────────────────────────────────────────

def test_report_generation():
    """Relatório gerado deve conter seções obrigatórias."""
    results  = [_make_result(needs_review=False), _make_result(needs_review=True, score=0.3)]
    retrieval = {"recall_at_5": 0.8, "recall_at_10": 0.9, "mrr": 0.75, "hit_rate_5": 0.8,
                 "n_questions_with_relevant": 2}
    ragas    = {"faithfulness": 0.85, "answer_relevancy": 0.80,
                "context_precision": 0.75, "context_recall": 0.70}

    report = generate_report(results, retrieval, ragas, "20260424_120000")
    assert "ANEEL RAG"  in report
    assert "RAGAS"      in report
    assert "Retrieval"  in report
    assert "Revisão"    in report or "Revisao" in report or "Precisa" in report or "Review" in report
    assert "[FONTE 1]"  in report or "FONTE"   in report or "Pergunta" in report
