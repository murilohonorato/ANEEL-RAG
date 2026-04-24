"""
07_evaluate.py
--------------
Avaliação completa do pipeline RAG com golden set + RAGAS.

Fluxo:
1. Para cada pergunta do golden set:
   a. Roda o pipeline completo (embedding + search + rerank)
   b. Gera resposta com GPT-4o
   c. Juiz leve (GPT-4o-mini) avalia faithfulness em 1 chamada
   d. Se score < threshold: regenera com prompt mais restritivo (até 3x)
   e. Após 3 falhas: marca como needs_review
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
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm

import importlib.util

from src.config import (
    COLLECTION_NAME,
    EVAL_LLM_MODEL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    OPENAI_API_KEY,
    QDRANT_PATH,
    RESULTS_DIR,
)
from src.utils.logger import logger
from src.utils.token_counter import count_tokens

# ── carregamento de 06_query.py (nome começa com dígito) ──────────────────────
def _load_query_module():
    spec = importlib.util.spec_from_file_location(
        "query_module",
        Path(__file__).parent / "06_query.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_query_mod = _load_query_module()
process_query          = _query_mod.process_query
embed_query            = _query_mod.embed_query
hybrid_search          = _query_mod.hybrid_search
reciprocal_rank_fusion = _query_mod.reciprocal_rank_fusion
lookup_parents         = _query_mod.lookup_parents
rerank                 = _query_mod.rerank
format_context         = _query_mod.format_context

# ── CONFIG ─────────────────────────────────────────────────────────────────────
GOLDEN_SET_PATH    = Path(__file__).parent.parent / "tests" / "golden_set.json"
JUDGE_THRESHOLD    = 0.5    # score mínimo do juiz para aceitar a resposta
MAX_RETRIES        = 3      # máximo de tentativas de regeneração
RAGAS_BATCH_SIZE   = 5      # perguntas por batch no RAGAS

# Prompts de regeneração progressivamente mais restritivos
_REGEN_PROMPTS = [
    # tentativa 1 — mais conservador
    (
        "Seja mais conservador. Responda APENAS o que está explicitamente escrito nas fontes. "
        "Não faça inferências. Cite cada afirmação com [TIPO NUMERO/ANO, Art. X]."
    ),
    # tentativa 2 — citação obrigatória por sentença
    (
        "ATENÇÃO: Cada sentença da sua resposta DEVE ter uma citação inline. "
        "Se não há fonte para uma informação, não a inclua. "
        "Formato obrigatório: [TIPO NUMERO/ANO, Art. X] após cada afirmação."
    ),
    # tentativa 3 — modo mínimo
    (
        "Responda em no máximo 3 frases, usando APENAS trechos literais das fontes. "
        "Cite cada trecho com [TIPO NUMERO/ANO, Art. X]. "
        "Se não encontrar a informação, diga: 'Não encontrei essa informação nas fontes disponíveis.'"
    ),
]


# ── Juiz leve (GPT-4o-mini, 1 chamada) ───────────────────────────────────────

def lightweight_judge(
    question: str,
    answer: str,
    contexts: list[str],
    client: OpenAI,
) -> dict:
    """
    Avalia faithfulness e presença de citações em uma única chamada GPT-4o-mini.
    Rápido (~0.5-1s). Usado no loop de autocorreção.

    Retorna:
        {"score": float 0-1, "issues": str, "has_citations": bool}
    """
    context_snippet = "\n---\n".join(c[:500] for c in contexts[:3])

    prompt = f"""Avalie a resposta abaixo em relação à pergunta e aos contextos fornecidos.

PERGUNTA: {question}

CONTEXTOS (resumo):
{context_snippet}

RESPOSTA GERADA:
{answer}

Avalie dois critérios:
1. FAITHFULNESS (0-1): A resposta é sustentada pelos contextos? Sem informações inventadas?
   - 1.0 = totalmente sustentada
   - 0.5 = parcialmente sustentada
   - 0.0 = não sustentada ou com alucinações

2. CITACOES: A resposta contém citações no formato [TIPO NUMERO/ANO, Art. X]?

Responda APENAS com JSON válido, sem markdown:
{{"score": <float>, "issues": "<descricao curta do problema ou 'ok'>", "has_citations": <true/false>}}"""

    try:
        response = client.chat.completions.create(
            model=EVAL_LLM_MODEL,
            temperature=0.0,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        # remover markdown e extrair apenas o bloco JSON
        raw = raw.replace("```json", "").replace("```", "").strip()
        # extrair primeiro objeto JSON da resposta via regex (mais robusto)
        json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group()
        result = json.loads(raw)
        return {
            "score":         float(result.get("score", 0.0)),
            "issues":        str(result.get("issues", "")),
            "has_citations": bool(result.get("has_citations", False)),
        }
    except Exception as exc:
        logger.warning(f"Juiz falhou: {exc}. Assumindo score=0.5")
        return {"score": 0.5, "issues": f"judge_error: {exc}", "has_citations": False}


# ── Geração com retry ─────────────────────────────────────────────────────────

def generate_with_retry(
    question: str,
    formatted_context: str,
    contexts: list[str],
    openai_client: OpenAI,
) -> dict:
    """
    Gera resposta com loop de autocorreção:
    - Tenta até MAX_RETRIES vezes se juiz reprovar
    - Cada tentativa usa prompt progressivamente mais restritivo
    - Após MAX_RETRIES falhas: marca needs_review=True

    Retorna:
        {
          "answer":       str,
          "attempts":     int,
          "final_score":  float,
          "needs_review": bool,
          "judge_history": list[dict],
        }
    """
    # LLM_MODEL e LLM_MAX_TOKENS já importados no topo do módulo

    SYSTEM_BASE = (
        "Você é um especialista em legislação do setor elétrico brasileiro. "
        "Responda APENAS com base nas fontes fornecidas. "
        "Cite SEMPRE as fontes no formato [TIPO NUMERO/ANO, Art. X]. "
        "Se a informação não estiver nas fontes, diga explicitamente que não encontrou. "
        "Responda em português formal."
    )

    judge_history = []
    answer        = ""

    for attempt in range(MAX_RETRIES):
        # montar prompt de geração
        extra = f"\n\nINSTRUCAO ADICIONAL: {_REGEN_PROMPTS[attempt - 1]}" if attempt > 0 else ""
        user_msg = (
            f"Fontes disponíveis:\n\n{formatted_context}\n\n---\n\n"
            f"Pergunta: {question}\n\n"
            f"Responda com base apenas nas fontes acima, "
            f"citando sempre no formato [TIPO NUMERO/ANO, Art. X].{extra}"
        )

        try:
            resp = openai_client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.1 if attempt == 0 else 0.0,
                max_tokens=LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_BASE},
                    {"role": "user",   "content": user_msg},
                ],
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"Erro na geração (tentativa {attempt+1}): {exc}")
            answer = "Erro na geração da resposta."
            judge_history.append({"attempt": attempt + 1, "score": 0.0, "issues": str(exc)})
            continue

        # avaliar com juiz leve
        judgment = lightweight_judge(question, answer, contexts, openai_client)
        judgment["attempt"] = attempt + 1
        judge_history.append(judgment)

        logger.debug(
            f"Tentativa {attempt+1}: score={judgment['score']:.2f} "
            f"citations={judgment['has_citations']} issues={judgment['issues']}"
        )

        if judgment["score"] >= JUDGE_THRESHOLD:
            return {
                "answer":        answer,
                "attempts":      attempt + 1,
                "final_score":   judgment["score"],
                "needs_review":  False,
                "judge_history": judge_history,
            }

    # esgotou tentativas → needs_review
    logger.warning(f"needs_review: pergunta nao atingiu threshold apos {MAX_RETRIES} tentativas")
    return {
        "answer":        answer,
        "attempts":      MAX_RETRIES,
        "final_score":   judge_history[-1]["score"] if judge_history else 0.0,
        "needs_review":  True,
        "judge_history": judge_history,
    }


# ── Pipeline por pergunta ─────────────────────────────────────────────────────

def run_question(
    item: dict,
    qdrant_client: QdrantClient,
    openai_client: OpenAI,
) -> dict:
    """
    Executa o pipeline completo para uma pergunta do golden set.
    Inclui o loop de autocorreção com juiz leve.
    """
    question = item["question"]
    t0       = time.time()

    # 1. Query processing
    processed = process_query(question)
    filters   = processed["filters"]

    # 2. Embedding
    vecs = embed_query(processed["clean"])

    # 3. Hybrid search
    dense_res, sparse_res = hybrid_search(qdrant_client, vecs["dense"], vecs["sparse"], filters)

    # 4. RRF
    rrf_results = reciprocal_rank_fusion(dense_res, sparse_res)

    # 5. Parent lookup
    contexts = lookup_parents(rrf_results)

    # 6. Reranking (só se houver contextos)
    if contexts:
        contexts = rerank(question, contexts)

    # 7. Formatar contexto
    formatted = format_context(contexts)
    context_texts  = [c["texto_parent"] for c in contexts]
    context_doc_ids = list({c["doc_id"] for c in contexts})

    elapsed_retrieval = round(time.time() - t0, 2)

    # 8. Geração com retry
    if not formatted.strip():
        gen_result = {
            "answer":        "Nao encontrei fontes relevantes.",
            "attempts":      0,
            "final_score":   0.0,
            "needs_review":  True,
            "judge_history": [],
        }
    else:
        gen_result = generate_with_retry(question, formatted, context_texts, openai_client)

    elapsed_total = round(time.time() - t0, 2)

    return {
        "question":          question,
        "expected_answer":   item.get("expected_answer", ""),
        "relevant_doc_ids":  item.get("relevant_doc_ids", []),
        "notes":             item.get("notes", ""),
        "generated_answer":  gen_result["answer"],
        "retrieved_contexts": context_texts,
        "retrieved_doc_ids": context_doc_ids,
        "attempts":          gen_result["attempts"],
        "final_judge_score": gen_result["final_score"],
        "needs_review":      gen_result["needs_review"],
        "judge_history":     gen_result["judge_history"],
        "timing": {
            "retrieval_s": elapsed_retrieval,
            "total_s":     elapsed_total,
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
            "question":          [r["question"]          for r in results],
            "answer":            [r["generated_answer"]  for r in results],
            "contexts":          [r["retrieved_contexts"] for r in results],
            "ground_truth":      [r["expected_answer"]   for r in results],
        }

        dataset = Dataset.from_dict(data)
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
    avg_attempts = sum(r["attempts"] for r in results) / max(len(results), 1)
    avg_time     = sum(r["timing"]["total_s"] for r in results) / max(len(results), 1)

    lines = [
        f"# ANEEL RAG — Relatório de Avaliação",
        f"",
        f"**Data:** {timestamp}  ",
        f"**Perguntas avaliadas:** {len(results)}  ",
        f"**Needs review:** {len(needs_review)}  ",
        f"**Score médio do juiz:** {avg_score:.2f}  ",
        f"**Tentativas médias:** {avg_attempts:.1f}  ",
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
            f"**Tentativas:** {r['attempts']}  ",
            f"**Resposta gerada:** {r['generated_answer'][:200]}...  ",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## Resultados por Pergunta",
        f"",
        f"| # | Pergunta | Score | Tentativas | Needs Review |",
        f"|---|----------|-------|------------|--------------|",
    ]

    for i, r in enumerate(results, 1):
        q        = r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"]
        review   = "⚠️" if r["needs_review"] else "✅"
        lines.append(
            f"| {i} | {q} | {r['final_judge_score']:.2f} | "
            f"{r['attempts']} | {review} |"
        )

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

    # carregar golden set
    if not golden_set_path.exists():
        raise FileNotFoundError(f"Golden set nao encontrado: {golden_set_path}")
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)
    if not golden_set:
        raise ValueError("Golden set esta vazio")
    logger.info(f"Golden set carregado: {len(golden_set)} perguntas")

    # inicializar clientes
    qdrant  = QdrantClient(path=str(QDRANT_PATH))
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # executar pipeline para cada pergunta
    results = []
    needs_review_count = 0

    for item in tqdm(golden_set, desc="Avaliando"):
        try:
            result = run_question(item, qdrant, openai_client)
            results.append(result)

            status = "⚠️ needs_review" if result["needs_review"] else f"✅ score={result['final_judge_score']:.2f}"
            logger.info(f"[{result['attempts']} tentativas] {status} — {item['question'][:60]}")

            if result["needs_review"]:
                needs_review_count += 1

        except Exception as exc:
            logger.error(f"Erro na pergunta '{item['question'][:50]}': {exc}")
            results.append({
                "question":          item["question"],
                "expected_answer":   item.get("expected_answer", ""),
                "relevant_doc_ids":  item.get("relevant_doc_ids", []),
                "notes":             item.get("notes", ""),
                "generated_answer":  f"ERRO: {exc}",
                "retrieved_contexts": [],
                "retrieved_doc_ids": [],
                "attempts":          0,
                "final_judge_score": 0.0,
                "needs_review":      True,
                "judge_history":     [],
                "timing":            {"retrieval_s": 0, "total_s": 0},
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
    ragas_metrics = {}
    if not skip_ragas:
        logger.info("Iniciando avaliacao RAGAS completa...")
        ragas_metrics = run_ragas_evaluation(results, openai_client)
        logger.info(f"RAGAS: {ragas_metrics}")
    else:
        ragas_metrics = {"note": "skipped"}

    # relatório
    report = generate_report(results, retrieval_metrics, ragas_metrics, timestamp)
    report_path = RESULTS_DIR / f"eval_report_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Relatorio salvo: {report_path}")

    # resumo final
    summary = {
        "timestamp":         timestamp,
        "n_questions":       len(results),
        "needs_review":      needs_review_count,
        "avg_judge_score":   round(sum(r["final_judge_score"] for r in results) / max(len(results), 1), 3),
        "avg_attempts":      round(sum(r["attempts"] for r in results) / max(len(results), 1), 2),
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
    parser.add_argument("--golden",      type=str, default=str(GOLDEN_SET_PATH),
                        help="Caminho para o golden set JSON")
    parser.add_argument("--skip-ragas",  action="store_true",
                        help="Pular avaliacao RAGAS completa (mais rapido)")
    args = parser.parse_args()

    summary = run_evaluation(
        golden_set_path=Path(args.golden),
        skip_ragas=args.skip_ragas,
    )
    print(f"\nAvaliacao concluida!")
    print(f"  Perguntas:      {summary['n_questions']}")
    print(f"  Needs review:   {summary['needs_review']}")
    print(f"  Score medio:    {summary['avg_judge_score']}")
    print(f"  Tentativas med: {summary['avg_attempts']}")
    print(f"  Relatorio:      {summary['report_path']}")
