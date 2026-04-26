"""
07_evaluate.py
--------------
Avaliação completa do pipeline RAG com golden set + RAGAS.

Fluxo:
1. Para cada pergunta do golden set:
   a. Executa query_pipeline() do módulo 6 (mesmo código de produção)
   b. Usa o resultado do crítico embutido no pipeline como sinal de needs_review
2. Métricas de retrieval diretas: Recall@K, MRR, Hit Rate
3. Avaliação RAGAS completa (4 métricas) sobre o batch final
4. Relatório em Markdown

Uso:
    python src/07_evaluate.py
    python src/07_evaluate.py --golden tests/golden_set.json
    python src/07_evaluate.py --skip-ragas   # apenas retrieval metrics
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm

from src.config import (
    EVAL_LLM_MODEL,
    OPENAI_API_KEY,
    QDRANT_PATH,
    RESULTS_DIR,
)
from src.utils.logger import logger

# ── carregamento de 06_query.py (nome começa com dígito) ──────────────────────
def _load_query_module():
    spec = importlib.util.spec_from_file_location(
        "query_module",
        Path(__file__).parent / "06_query.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_query_mod     = _load_query_module()
query_pipeline = _query_mod.query_pipeline

# ── CONFIG ─────────────────────────────────────────────────────────────────────
GOLDEN_SET_PATH  = Path(__file__).parent.parent / "tests" / "golden_set.json"
RAGAS_BATCH_SIZE = 5


# ── Pipeline por pergunta ─────────────────────────────────────────────────────

def run_question(
    item: dict,
    qdrant_client: QdrantClient,
) -> dict:
    """
    Executa query_pipeline() do módulo 6 para uma pergunta do golden set.
    Usa o resultado do crítico embutido como sinal de qualidade.
    """
    question = item["question"]
    t0       = time.time()

    result   = query_pipeline(question, client=qdrant_client)

    contexts         = result["contexts"]
    context_texts    = [c["texto_parent"] for c in contexts]
    context_doc_ids  = list({c["doc_id"] for c in contexts})

    critic       = result.get("critic")
    is_abstention = result["answer"].startswith("Nao encontrei")

    # needs_review: sem fontes, abstinência forçada ou crítico rejeitou
    needs_review  = not contexts or is_abstention or (critic is not None and not critic["valid"])
    judge_score   = 0.0 if is_abstention or not contexts else (1.0 if not critic or critic["valid"] else 0.5)
    judge_history = [{"attempt": 1, **critic}] if critic else []

    timing = result.get("timing", {})

    return {
        "question":           question,
        "expected_answer":    item.get("expected_answer", ""),
        "relevant_doc_ids":   item.get("relevant_doc_ids", []),
        "notes":              item.get("notes", ""),
        "generated_answer":   result["answer"],
        "retrieved_contexts": context_texts,
        "retrieved_doc_ids":  context_doc_ids,
        "attempts":           1,
        "final_judge_score":  judge_score,
        "needs_review":       needs_review,
        "judge_history":      judge_history,
        "timing": {
            "retrieval_s": round(timing.get("hybrid_search", 0), 2),
            "total_s":     round(time.time() - t0, 2),
        },
    }


# ── Métricas de retrieval ─────────────────────────────────────────────────────

def compute_retrieval_metrics(results: list[dict]) -> dict:
    """
    Calcula Recall@K, MRR e Hit Rate para os resultados.
    Só é calculado para questões que têm relevant_doc_ids preenchido.
    """
    filtered = [r for r in results if r.get("relevant_doc_ids")]
    if not filtered:
        return {"note": "Nenhuma questao com relevant_doc_ids preenchido"}

    def recall_at_k(k: int) -> float:
        hits = 0
        for r in filtered:
            retrieved = set(r["retrieved_doc_ids"][:k])
            relevant  = set(r["relevant_doc_ids"])
            if retrieved & relevant:
                hits += 1
        return round(hits / len(filtered), 3)

    def mrr() -> float:
        rr_sum = 0.0
        for r in filtered:
            relevant = set(r["relevant_doc_ids"])
            for rank, doc_id in enumerate(r["retrieved_doc_ids"], 1):
                if doc_id in relevant:
                    rr_sum += 1.0 / rank
                    break
        return round(rr_sum / len(filtered), 3)

    return {
        "n_questions_with_relevant": len(filtered),
        "recall_at_5":  recall_at_k(5),
        "recall_at_10": recall_at_k(10),
        "recall_at_20": recall_at_k(20),
        "mrr":          mrr(),
        "hit_rate_5":   recall_at_k(5),
    }


# ── Avaliação RAGAS ───────────────────────────────────────────────────────────

def run_ragas_evaluation(results: list[dict], openai_client: OpenAI) -> dict:
    """
    Avaliação RAGAS completa com GPT-4o-mini como juiz.
    Roda uma vez no final do batch.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        from datasets import Dataset

        llm = ChatOpenAI(model=EVAL_LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)

        data = {
            "question":     [r["question"]          for r in results],
            "answer":       [r["generated_answer"]  for r in results],
            "contexts":     [r["retrieved_contexts"] for r in results],
            "ground_truth": [r["expected_answer"]   for r in results],
        }

        dataset      = Dataset.from_dict(data)
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=LangchainLLMWrapper(llm),
        )

        return {
            "faithfulness":      round(float(ragas_result["faithfulness"]),      3),
            "answer_relevancy":  round(float(ragas_result["answer_relevancy"]),  3),
            "context_precision": round(float(ragas_result["context_precision"]), 3),
            "context_recall":    round(float(ragas_result["context_recall"]),    3),
        }

    except ImportError:
        logger.warning("ragas/langchain nao instalado — pulando avaliacao RAGAS completa")
        return {"error": "ragas not installed"}
    except Exception as exc:
        logger.error(f"Erro no RAGAS: {exc}")
        return {"error": str(exc)}


# ── Relatório Markdown ────────────────────────────────────────────────────────

def generate_report(
    results: list[dict],
    retrieval_metrics: dict,
    ragas_metrics: dict,
    timestamp: str,
) -> str:
    """Gera relatório completo em Markdown."""

    needs_review = [r for r in results if r["needs_review"]]
    avg_score    = sum(r["final_judge_score"] for r in results) / max(len(results), 1)
    avg_time     = sum(r["timing"]["total_s"] for r in results) / max(len(results), 1)

    lines = [
        f"# ANEEL RAG — Relatório de Avaliação",
        f"",
        f"**Data:** {timestamp}  ",
        f"**Perguntas avaliadas:** {len(results)}  ",
        f"**Needs review:** {len(needs_review)}  ",
        f"**Score médio do crítico:** {avg_score:.2f}  ",
        f"**Tempo médio por pergunta:** {avg_time:.1f}s  ",
        f"",
        f"---",
        f"",
        f"## Métricas RAGAS",
        f"",
        f"| Métrica | Score |",
        f"|---------|-------|",
    ]

    if "error" in ragas_metrics:
        lines.append(f"| Erro | {ragas_metrics['error']} |")
    else:
        for k, v in ragas_metrics.items():
            lines.append(f"| {k} | {v:.3f} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Métricas de Retrieval",
        f"",
        f"| Métrica | Valor |",
        f"|---------|-------|",
    ]

    for k, v in retrieval_metrics.items():
        if k != "note":
            lines.append(f"| {k} | {v} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Perguntas que Precisam de Revisão ({len(needs_review)})",
        f"",
    ]

    for r in needs_review:
        lines += [
            f"### ❌ {r['question']}",
            f"**Score final:** {r['final_judge_score']:.2f}  ",
            f"**Resposta gerada:** {r['generated_answer'][:200]}...  ",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## Resultados por Pergunta",
        f"",
        f"| # | Pergunta | Score | Needs Review |",
        f"|---|----------|-------|--------------|",
    ]

    for i, r in enumerate(results, 1):
        q      = r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"]
        review = "⚠️" if r["needs_review"] else "✅"
        lines.append(f"| {i} | {q} | {r['final_judge_score']:.2f} | {review} |")

    return "\n".join(lines)


# ── Pipeline batch principal ──────────────────────────────────────────────────

def run_evaluation(
    golden_set_path: Path = GOLDEN_SET_PATH,
    skip_ragas: bool = False,
) -> dict:
    """
    Executa avaliação completa sobre o golden set.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not golden_set_path.exists():
        raise FileNotFoundError(f"Golden set nao encontrado: {golden_set_path}")
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)
    if not golden_set:
        raise ValueError("Golden set esta vazio")
    logger.info(f"Golden set carregado: {len(golden_set)} perguntas")

    qdrant        = QdrantClient(path=str(QDRANT_PATH))
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    results            = []
    needs_review_count = 0

    for item in tqdm(golden_set, desc="Avaliando"):
        try:
            result = run_question(item, qdrant)
            results.append(result)

            status = "⚠️ needs_review" if result["needs_review"] else f"✅ score={result['final_judge_score']:.2f}"
            logger.info(f"{status} — {item['question'][:60]}")

            if result["needs_review"]:
                needs_review_count += 1

        except Exception as exc:
            logger.error(f"Erro na pergunta '{item['question'][:50]}': {exc}")
            results.append({
                "question":           item["question"],
                "expected_answer":    item.get("expected_answer", ""),
                "relevant_doc_ids":   item.get("relevant_doc_ids", []),
                "notes":              item.get("notes", ""),
                "generated_answer":   f"ERRO: {exc}",
                "retrieved_contexts": [],
                "retrieved_doc_ids":  [],
                "attempts":           0,
                "final_judge_score":  0.0,
                "needs_review":       True,
                "judge_history":      [],
                "timing":             {"retrieval_s": 0, "total_s": 0},
            })

    # salvar resultados brutos
    raw_path = RESULTS_DIR / f"eval_run_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Resultados salvos: {raw_path}")

    # métricas de retrieval
    retrieval_metrics = compute_retrieval_metrics(results)
    logger.info(f"Retrieval metrics: {retrieval_metrics}")

    # RAGAS completo
    if not skip_ragas:
        logger.info("Iniciando avaliacao RAGAS completa...")
        ragas_metrics = run_ragas_evaluation(results, openai_client)
        logger.info(f"RAGAS: {ragas_metrics}")
    else:
        ragas_metrics = {"note": "skipped"}

    # relatório
    report      = generate_report(results, retrieval_metrics, ragas_metrics, timestamp)
    report_path = RESULTS_DIR / f"eval_report_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Relatorio salvo: {report_path}")

    summary = {
        "timestamp":         timestamp,
        "n_questions":       len(results),
        "needs_review":      needs_review_count,
        "avg_judge_score":   round(sum(r["final_judge_score"] for r in results) / max(len(results), 1), 3),
        "retrieval_metrics": retrieval_metrics,
        "ragas_metrics":     ragas_metrics,
        "raw_path":          str(raw_path),
        "report_path":       str(report_path),
    }
    logger.info(f"Avaliacao concluida: {summary}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modulo 7 — Avaliacao RAG com RAGAS")
    parser.add_argument("--golden",     type=str, default=str(GOLDEN_SET_PATH),
                        help="Caminho para o golden set JSON")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Pular avaliacao RAGAS completa (mais rapido)")
    args = parser.parse_args()

    summary = run_evaluation(
        golden_set_path=Path(args.golden),
        skip_ragas=args.skip_ragas,
    )
    print(f"\nAvaliacao concluida!")
    print(f"  Perguntas:    {summary['n_questions']}")
    print(f"  Needs review: {summary['needs_review']}")
    print(f"  Score medio:  {summary['avg_judge_score']}")
    print(f"  Relatorio:    {summary['report_path']}")
