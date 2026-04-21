"""
06_query.py
-----------
Pipeline completo de consulta RAG.

Fluxo:
1. Query Processing — extrai filtros (ano, tipo, nº) + reescreve query
2. Embedding — bge-m3 gera dense_vec + sparse_vec
3. Hybrid Search — Qdrant (dense + sparse + filtros de metadados)
4. Fusão RRF — combina rankings, boost em ementas
5. Parent Lookup — recupera texto_parent do payload para cada child
6. Reranking — bge-reranker-v2-m3 reordena top-20 → top-5
7. Geração — Claude Sonnet com instrução de citação

Uso:
    python src/06_query.py "qual o prazo para revisão tarifária?"
    python src/06_query.py --interactive   # modo chat
"""

import os
# CONFIG
QDRANT_PATH    = "qdrant_db"
COLLECTION     = "aneel_legislacao"
EMBED_MODEL    = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL      = "claude-sonnet-4-6"

RETRIEVAL_K    = 50    # candidatos por busca (dense e sparse)
RERANK_TOP_N   = 5     # chunks finais para o LLM

SYSTEM_PROMPT = """Você é um especialista em legislação do setor elétrico brasileiro.
Responda APENAS com base nos trechos fornecidos.
Cite obrigatoriamente a fonte de cada afirmação no formato [TIPO NUMERO/ANO, Art. X].
Se a informação não estiver nos trechos, diga explicitamente que não encontrou."""

# TODO: implementar com Claude Code
