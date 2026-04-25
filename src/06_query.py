"""
06_query.py
-----------
Pipeline completo de consulta RAG para legislação ANEEL.

Fluxo por pergunta:
1. Query Processing  — extrai filtros (ano, tipo, número) + limpa a query
2. Embedding         — bge-m3 gera dense_vec + sparse_vec da query
3. Hybrid Search     — Qdrant busca com dense + sparse + filtros de metadados
4. Fusão RRF         — combina rankings, boost de 1.3x em ementas
5. Parent Lookup     — recupera texto_parent do payload para cada child
6. Reranking         — bge-reranker-v2-m3 reordena top-20 → top-5
7. Geração           — GPT-4o responde com instrução de citação obrigatória

Uso:
    python src/06_query.py
    python src/06_query.py --question "O que é microgeração distribuída?"
    python src/06_query.py --question "..." --debug
    python src/06_query.py --question "..." --no-rerank
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from src.config import (
    COLLECTION_NAME,
    CONTEXT_MAX_TOKENS,
    CRITIC_MODEL,
    EMENTA_BOOST,
    HYBRID_TOP_K,
    LLM_MODEL,
    OPENAI_API_KEY,
    QDRANT_PATH,
    RERANKER_MODEL,
    RERANK_TOP_N,
    RRF_K,
    VALIDATION_MAX_CYCLES,
)
from src.utils.logger import logger
from src.utils.qdrant_filters import build_filter
from src.utils.token_counter import count_tokens, truncate_to_tokens

# ── CONFIG ─────────────────────────────────────────────────────────────────────
from src.config import LLM_MAX_TOKENS, LLM_TEMPERATURE  # centralizados em config.py
LLM_RETRY_MAX = 2

SYSTEM_PROMPT = """Você é um especialista em legislação do setor elétrico brasileiro, \
com amplo conhecimento sobre normas da ANEEL (Agência Nacional de Energia Elétrica).

Regras obrigatórias:
1. Responda APENAS com base nas fontes fornecidas abaixo.
2. Cite SEMPRE as fontes inline no formato [TIPO NUMERO/ANO, Art. X] — exemplo: [REN 1000/2021, Art. 3º].
3. Se a informação não estiver nas fontes, diga explicitamente: "Não encontrei essa informação nas fontes disponíveis."
4. Responda em português formal e objetivo.
5. Não invente informações, datas, números ou artigos que não apareçam nas fontes."""

CRITIC_SYSTEM_PROMPT = """\
Você é um avaliador crítico de respostas de um sistema RAG jurídico.
Analise se a resposta é fiel às fontes e responde corretamente à pergunta.

Responda SOMENTE com JSON no formato: {"valid": true/false, "reason": "explicação em uma frase"}

Critérios de validade (todos devem ser atendidos):
1. A resposta está fundamentada nas fontes fornecidas (sem informações inventadas)?
2. Se a pergunta menciona uma entidade específica (usina, empresa, pessoa), a resposta trata DESSA entidade, não de outra?
3. A resposta não é tão genérica que poderia se aplicar a qualquer entidade similar?

Exceção: se a resposta diz "Não encontrei essa informação nas fontes disponíveis", marque valid=true.\
"""


# ── 6.1 — Query Processing ────────────────────────────────────────────────────

_STOP_WORDS = re.compile(
    r'\b(a|o|as|os|de|da|do|das|dos|que|qual|quais|e|em|no|na|nos|nas|'
    r'me|diga|fale|sobre|como|quando|onde|por|para|com|se|'
    r'resolucao|norma|lei|decreto|portaria|despacho)\b',
    re.IGNORECASE,
)

_ANO_RE    = re.compile(r'\b(2016|2021|2022)\b')
_TIPO_RE   = re.compile(r'\b(REN|REH|REA|DSP|PRT)\b', re.IGNORECASE)
_NUMERO_RE = re.compile(r'\b(?:REN|REH|REA|DSP|PRT)\s*[n°]*\s*(\d+)\b', re.IGNORECASE)

# Extração de entidade: usinas (PCH/EOL/etc.) e empresas (S.A./Ltda.)
_PLANT_RE = re.compile(
    r'\b(PCH|EOL|UHE|CGH|UFV|UTE|PCG|CGEI)\s+((?:\w+\s+){0,3}\w+)',
    re.UNICODE | re.IGNORECASE,
)
_EMPRESA_RE = re.compile(
    r'\b([A-ZÀ-Ú]\w+(?:\s+(?:[A-ZÀ-Ú]\w*|\w*[IVX]+))+?)\s+(?:S\.A\.|Ltda\.?|EIRELI|S/A)',
    re.UNICODE,
)


def _extract_entity_name(question: str) -> Optional[str]:
    """Extrai nome de usina (PCH/EOL/etc.) ou empresa (S.A./Ltda.) da pergunta."""
    m = _PLANT_RE.search(question)
    if m:
        return f"{m.group(1).upper()} {m.group(2).strip()}"
    m = _EMPRESA_RE.search(question)
    if m:
        return m.group(1).strip()
    return None


def process_query(question: str) -> dict:
    """
    Extrai filtros e limpa a query para melhor retrieval.

    Retorna:
        {
          "original": str,
          "clean":    str,
          "filters":  {"ano": int|None, "tipo_sigla": str|None, "numero": int|None}
        }
    """
    filters: dict = {"ano": None, "tipo_sigla": None, "numero": None, "entity_name": None}

    m_ano = _ANO_RE.search(question)
    if m_ano:
        filters["ano"] = int(m_ano.group())

    m_tipo = _TIPO_RE.search(question)
    if m_tipo:
        filters["tipo_sigla"] = m_tipo.group().upper()

    m_num = _NUMERO_RE.search(question)
    if m_num:
        filters["numero"] = int(m_num.group(1))

    filters["entity_name"] = _extract_entity_name(question)

    # limpeza da query
    clean = _STOP_WORDS.sub(" ", question)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if not clean:
        clean = question  # fallback: usar original se ficou vazia

    return {"original": question, "clean": clean, "filters": filters}


# ── 6.2 — Embedding da query ──────────────────────────────────────────────────

_embedding_model = None


def _get_embedding_model():
    """Carrega bge-m3 na primeira chamada (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from FlagEmbedding import BGEM3FlagModel
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Carregando bge-m3 no device={device}...")
        _embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=(device != "cpu"),
            device=device,
        )
        logger.info("bge-m3 carregado.")
    return _embedding_model


def embed_query(query_text: str) -> dict:
    """
    Gera dense_vec + sparse_vec para a query.
    Retorna {"dense": list[float], "sparse": SparseVector}.
    """
    model  = _get_embedding_model()
    result = model.encode(
        [query_text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vec  = result["dense_vecs"][0].tolist()
    sparse_raw = result["lexical_weights"][0]  # {token_id: weight}
    indices    = list(sparse_raw.keys())
    values     = [float(sparse_raw[i]) for i in indices]
    sparse_vec = SparseVector(indices=indices, values=values)

    return {"dense": dense_vec, "sparse": sparse_vec}


# ── 6.3 — Hybrid Search ───────────────────────────────────────────────────────

def _search_dense(client: QdrantClient, dense_vec: list, qdrant_filter, top_k: int):
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vec,
        using="dense",
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
        score_threshold=0.1,
    )
    return result.points


def _search_sparse(client: QdrantClient, sparse_vec: SparseVector, qdrant_filter, top_k: int):
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vec,
        using="sparse",
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )
    return result.points


def hybrid_search(
    client: QdrantClient,
    dense_vec: list,
    sparse_vec: SparseVector,
    filters: dict,
    top_k: int = HYBRID_TOP_K,
) -> tuple[list, list]:
    """
    Executa busca densa e esparsa em paralelo no Qdrant.
    Retorna (dense_results, sparse_results).
    """
    # apenas filtros indexados no Qdrant
    qdrant_filters = {k: v for k, v in filters.items() if k in ("ano", "tipo_sigla") and v}
    qdrant_filter  = build_filter(qdrant_filters)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_dense  = ex.submit(_search_dense,  client, dense_vec,  qdrant_filter, top_k)
        f_sparse = ex.submit(_search_sparse, client, sparse_vec, qdrant_filter, top_k)
        dense_results  = f_dense.result()
        sparse_results = f_sparse.result()

    return dense_results, sparse_results


# ── 6.4 — Fusão RRF ──────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = RRF_K,
    ementa_boost: float = EMENTA_BOOST,
) -> list[dict]:
    """
    Combina rankings dense e sparse via RRF.
    Ementas recebem boost multiplicativo no score.
    Retorna lista de dicts ordenada por score RRF decrescente.
    """
    scores:   dict[str, float] = {}
    payloads: dict[str, dict]  = {}

    for rank, hit in enumerate(dense_results):
        pid = str(hit.id)
        scores[pid]   = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        payloads[pid] = hit.payload or {}

    for rank, hit in enumerate(sparse_results):
        pid = str(hit.id)
        scores[pid]   = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        if pid not in payloads:
            payloads[pid] = hit.payload or {}

    # boost em ementas
    for pid, payload in payloads.items():
        if payload.get("is_ementa") or payload.get("chunk_type") == "ementa":
            scores[pid] *= ementa_boost

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"id": pid, "rrf_score": score, "payload": payloads[pid]}
        for pid, score in ranked
    ]


def boost_by_entity(rrf_results: list[dict], entity_name: Optional[str]) -> list[dict]:
    """
    Re-ordena resultados RRF colocando primeiro os que mencionam a entidade extraída.
    Preserva a ordem relativa dentro de cada grupo (match / não-match).
    """
    if not entity_name:
        return rrf_results

    needle = entity_name.lower()
    matched, unmatched = [], []
    for hit in rrf_results:
        p = hit.get("payload", {})
        haystack = (
            (p.get("texto") or "") + " " +
            (p.get("ementa") or "") + " " +
            (p.get("texto_parent") or "")
        ).lower()
        (matched if needle in haystack else unmatched).append(hit)
    return matched + unmatched


# ── 6.5 — Parent Lookup ───────────────────────────────────────────────────────

def lookup_parents(rrf_results: list[dict]) -> list[dict]:
    """
    Para cada resultado RRF:
    - child  → usa texto_parent do payload (artigo completo)
    - ementa/fallback/table → usa o próprio texto como contexto

    Deduplica por parent_id e retorna até 20 contextos únicos.
    """
    seen:     set[str]  = set()
    contexts: list[dict] = []

    for hit in rrf_results:
        payload    = hit["payload"]
        chunk_type = payload.get("chunk_type", "")
        parent_id  = payload.get("parent_id") or hit["id"]
        doc_id     = payload.get("doc_id", "")

        dedup_key = f"{doc_id}__{parent_id}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        context_text = (
            payload.get("texto_parent") or payload.get("texto", "")
            if chunk_type == "child"
            else payload.get("texto", "")
        )

        if not context_text:
            continue

        contexts.append({
            "texto_parent": context_text,
            "doc_id":     doc_id,
            "tipo_sigla": payload.get("tipo_sigla", ""),
            "numero":     payload.get("numero"),
            "ano":        payload.get("ano"),
            "data_pub":   payload.get("data_pub", ""),
            "situacao":   payload.get("situacao", ""),
            "ementa":     payload.get("ementa", ""),
            "chunk_type": chunk_type,
            "parent_id":  parent_id,
            "rrf_score":  hit["rrf_score"],
        })

        if len(contexts) >= 20:
            break

    return contexts


# ── 6.6 — Reranking ───────────────────────────────────────────────────────────

_reranker_model = None


def _get_reranker():
    """Carrega bge-reranker na primeira chamada (singleton)."""
    global _reranker_model
    if _reranker_model is None:
        from FlagEmbedding import FlagReranker
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Carregando bge-reranker no device={device}...")
        _reranker_model = FlagReranker(
            RERANKER_MODEL,
            use_fp16=(device != "cpu"),
            device=device,
        )
        logger.info("bge-reranker carregado.")
    return _reranker_model


def rerank(question: str, contexts: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Reordena contextos usando cross-encoder bge-reranker-v2-m3.
    Retorna top_n contextos com maior score de relevância.
    """
    if not contexts:
        return []

    reranker = _get_reranker()
    pairs    = [(question, ctx["texto_parent"]) for ctx in contexts]
    scores   = reranker.compute_score(pairs, normalize=True)

    for ctx, score in zip(contexts, scores):
        ctx["rerank_score"] = float(score)

    ranked = sorted(contexts, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]


# ── 6.7 — Formatação do contexto ─────────────────────────────────────────────

def format_context(contexts: list[dict]) -> str:
    """
    Formata contextos para o prompt do LLM respeitando CONTEXT_MAX_TOKENS.
    """
    parts  = []
    tokens = 0

    for i, ctx in enumerate(contexts, 1):
        tipo    = ctx.get("tipo_sigla", "")
        numero  = ctx.get("numero")
        ano     = ctx.get("ano", "")
        num_str = str(int(numero)) if numero and str(numero) != "nan" else "s/n"
        situacao = ctx.get("situacao", "")
        data_pub = ctx.get("data_pub", "")

        header  = f"[FONTE {i}] {tipo} {num_str}/{ano}"
        meta    = f"Data: {data_pub} | Situacao: {situacao}"
        body    = ctx["texto_parent"]
        section = f"{header}\n{meta}\n{body}\n---"

        section_tokens = count_tokens(section)
        if tokens + section_tokens > CONTEXT_MAX_TOKENS:
            budget = CONTEXT_MAX_TOKENS - tokens - count_tokens(f"{header}\n{meta}\n\n---")
            if budget > 50:
                body    = truncate_to_tokens(body, budget)
                section = f"{header}\n{meta}\n{body}\n---"
            else:
                break

        parts.append(section)
        tokens += count_tokens(section)

    return "\n\n".join(parts)


# ── 6.8 — Geração com GPT-4o ─────────────────────────────────────────────────

def generate_answer(question: str, context: str, debug: bool = False) -> str:
    """
    Gera resposta usando GPT-4o com instrução de citação obrigatória.
    Retry automático (até LLM_RETRY_MAX vezes) em caso de erro de API.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    user_message = (
        f"Fontes disponíveis:\n\n{context}\n\n---\n\n"
        f"Pergunta: {question}\n\n"
        "Responda com base apenas nas fontes acima, "
        "citando sempre no formato [TIPO NUMERO/ANO, Art. X]."
    )

    for attempt in range(LLM_RETRY_MAX + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            answer = response.choices[0].message.content.strip()
            if debug:
                u = response.usage
                logger.debug(
                    f"GPT-4o tokens — prompt: {u.prompt_tokens}, "
                    f"completion: {u.completion_tokens}"
                )
            return answer

        except Exception as exc:
            if attempt < LLM_RETRY_MAX:
                wait = 2 ** attempt
                logger.warning(
                    f"Erro OpenAI (tentativa {attempt+1}): {exc}. Aguardando {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"Falha apos {LLM_RETRY_MAX+1} tentativas: {exc}")
                raise


# ── 6.9 — Validação pelo crítico ─────────────────────────────────────────────

def critic_check(
    question: str,
    answer: str,
    context: str,
    entity_name: Optional[str],
) -> dict:
    """
    Valida a resposta usando gpt-4o-mini como crítico.
    Retorna {"valid": bool, "reason": str}.
    Em caso de erro de API, aceita a resposta (fail-open).
    """
    entity_hint = f"\nEntidade específica perguntada: {entity_name}" if entity_name else ""
    user_msg = (
        f"Pergunta: {question}{entity_hint}\n\n"
        f"Fontes utilizadas (trecho):\n{context[:3000]}\n\n"
        f"Resposta gerada:\n{answer}"
    )
    try:
        oai = OpenAI(api_key=OPENAI_API_KEY)
        resp = oai.chat.completions.create(
            model=CRITIC_MODEL,
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return {"valid": bool(data.get("valid", True)), "reason": data.get("reason", "")}
    except Exception as exc:
        logger.warning(f"Crítico falhou ({exc}) — aceitando resposta por padrão")
        return {"valid": True, "reason": "critic_error"}


def _reformulate_query(clean_q: str, entity_name: Optional[str], reason: str) -> str:
    """Reformula a query para re-retrieval, enfatizando a entidade específica."""
    if entity_name:
        return f"{clean_q} especificamente {entity_name}"
    return clean_q


# ── 6.10 — Passe único do pipeline (interno) ──────────────────────────────────

def _pipeline_single_pass(
    question: str,
    search_query: str,
    filters: dict,
    entity_name: Optional[str],
    client: QdrantClient,
    debug: bool,
    use_rerank: bool,
) -> dict:
    """
    Executa um passe completo: embedding → hybrid search → RRF+boost → parent lookup
    → reranking → geração.
    Retorna {answer, contexts, formatted_context, timing}.
    """
    timing: dict = {}

    # Embedding
    t    = time.time()
    vecs = embed_query(search_query)
    timing["embedding"] = round(time.time() - t, 3)

    # Hybrid Search
    t = time.time()
    dense_res, sparse_res = hybrid_search(client, vecs["dense"], vecs["sparse"], filters)
    timing["hybrid_search"] = round(time.time() - t, 3)

    if debug:
        logger.debug(f"Dense hits: {len(dense_res)} | Sparse hits: {len(sparse_res)}")

    # RRF + Entity Boost
    t           = time.time()
    rrf_results = reciprocal_rank_fusion(dense_res, sparse_res)
    rrf_results = boost_by_entity(rrf_results, entity_name)
    timing["rrf"] = round(time.time() - t, 3)

    if debug:
        for r in rrf_results[:5]:
            logger.debug(
                f"RRF: {r['payload'].get('doc_id')} "
                f"[{r['payload'].get('chunk_type')}] score={r['rrf_score']:.4f}"
            )

    # Parent Lookup
    t        = time.time()
    contexts = lookup_parents(rrf_results)
    timing["parent_lookup"] = round(time.time() - t, 3)

    # Reranking
    if use_rerank and contexts:
        t        = time.time()
        contexts = rerank(question, contexts)
        timing["reranking"] = round(time.time() - t, 3)

        if debug:
            for ctx in contexts:
                logger.debug(
                    f"Rerank: {ctx['doc_id']} score={ctx.get('rerank_score', 0):.4f}"
                )
    else:
        contexts = contexts[:RERANK_TOP_N]
        timing["reranking"] = 0.0

    formatted = format_context(contexts)

    if not formatted.strip():
        timing["generation"] = 0.0
        return {
            "answer":            "Nao encontrei fontes relevantes para responder a esta pergunta.",
            "contexts":          [],
            "formatted_context": "",
            "timing":            timing,
        }

    t      = time.time()
    answer = generate_answer(question, formatted, debug=debug)
    timing["generation"] = round(time.time() - t, 3)

    return {
        "answer":            answer,
        "contexts":          contexts,
        "formatted_context": formatted,
        "timing":            timing,
    }


# ── Pipeline principal ────────────────────────────────────────────────────────

def query_pipeline(
    question: str,
    client: Optional[QdrantClient] = None,
    debug: bool = False,
    use_rerank: bool = True,
) -> dict:
    """
    Executa o pipeline completo para uma pergunta com loop gen/validação.

    Fluxo:
        query processing → (embedding → hybrid search → RRF+entity boost
        → parent lookup → reranking → geração) → crítico → [retry opcional]

    Retorna:
        {
          "question":  str,
          "answer":    str,
          "contexts":  list[dict],
          "filters":   dict,
          "timing":    dict,
          "critic":    dict | None,
        }
    """
    t0 = time.time()

    # 1. Query processing (inclui extração de entidade via regex)
    t         = time.time()
    processed = process_query(question)
    clean_q   = processed["clean"]
    filters   = processed["filters"]
    entity_name = filters.get("entity_name")
    timing: dict = {"query_processing": round(time.time() - t, 3)}

    if debug:
        logger.debug(f"Query limpa: '{clean_q}' | Filtros: {filters}")

    if client is None:
        client = QdrantClient(path=str(QDRANT_PATH))

    # Primeiro passe completo
    pass1    = _pipeline_single_pass(
        question, clean_q, filters, entity_name, client, debug, use_rerank
    )
    answer   = pass1["answer"]
    contexts = pass1["contexts"]
    timing.update(pass1["timing"])

    critic_result = None
    is_abstention = answer.startswith("Nao encontrei")

    # Validação pelo crítico (pula abstinências e ausência de contexto)
    if not is_abstention and pass1["formatted_context"] and VALIDATION_MAX_CYCLES >= 1:
        t = time.time()
        critic_result = critic_check(question, answer, pass1["formatted_context"], entity_name)
        timing["validation"] = round(time.time() - t, 3)

        if debug:
            logger.debug(f"Crítico: {critic_result}")

        if not critic_result["valid"]:
            logger.info(
                f"Crítico rejeitou (tentativa 1): {critic_result['reason']} "
                "— reformulando query e retentando"
            )
            new_q = _reformulate_query(clean_q, entity_name, critic_result["reason"])
            pass2 = _pipeline_single_pass(
                question, new_q, filters, entity_name, client, debug, use_rerank
            )
            answer   = pass2["answer"]
            contexts = pass2["contexts"]
            for k, v in pass2["timing"].items():
                timing[k] = timing.get(k, 0) + v
            critic_result = {"valid": True, "reason": "second_attempt_accepted"}

    timing["total"] = round(time.time() - t0, 3)

    return {
        "question": question,
        "answer":   answer,
        "contexts": contexts,
        "filters":  filters,
        "timing":   timing,
        "critic":   critic_result,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _interactive_loop(client: QdrantClient, debug: bool, use_rerank: bool) -> None:
    print("\nANEEL RAG — Pipeline de Consulta")
    print("Digite 'sair' para encerrar.\n")

    while True:
        try:
            question = input("Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not question:
            continue
        if question.lower() in ("sair", "exit", "quit"):
            break

        result = query_pipeline(question, client=client, debug=debug, use_rerank=use_rerank)
        print(f"\n{result['answer']}\n")

        if debug:
            print(f"Timing:  {result['timing']}")
            print(f"Filtros: {result['filters']}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modulo 6 — Pipeline de Consulta RAG")
    parser.add_argument("--question", "-q", type=str, help="Pergunta unica (modo nao-interativo)")
    parser.add_argument("--debug",     action="store_true", help="Mostrar scores e timing")
    parser.add_argument("--no-rerank", action="store_true", help="Pular reranking")
    args = parser.parse_args()

    qdrant = QdrantClient(path=str(QDRANT_PATH))

    if args.question:
        result = query_pipeline(
            args.question,
            client=qdrant,
            debug=args.debug,
            use_rerank=not args.no_rerank,
        )
        print(f"\n{result['answer']}\n")
        if args.debug:
            print(f"Timing:  {result['timing']}")
            print(f"Filtros: {result['filters']}")
    else:
        _interactive_loop(qdrant, debug=args.debug, use_rerank=not args.no_rerank)
