# Arquitetura do Sistema ANEEL RAG

## Visão Geral

Sistema de Retrieval-Augmented Generation (RAG) para responder perguntas sobre legislação do setor elétrico brasileiro. O dataset compreende documentos reais da ANEEL dos anos 2016, 2021 e 2022 (~18.688 documentos únicos).

---

## Pipeline Completo

```
[FASE OFFLINE — indexação]

JSON bruto (metadados ANEEL)
    ↓  Módulo 1 — consolidate_metadata.py
metadata.parquet (26.768 docs normalizados)
    ↓  Módulo 2 — download_pdfs.py (já executado)
PDFs em data/pdfs/{ano}/
    ↓  Módulo 3 — parse.py
texts/{doc_id}.txt (texto limpo, tabelas em Markdown)
    ↓  Módulo 4 — chunk.py
chunks.parquet (228.078 chunks)
    ↓  Módulo 5 — embed_index.py
Qdrant local — collection "aneel_legislacao"
    (dense 1024-dim + sparse BM25)

[FASE ONLINE — por consulta]

Pergunta do usuário
    ↓  Query Processing
Filtros extraídos (ano, tipo, número) + query limpa
    ↓  Embedding (bge-m3)
dense_vec [1024] + sparse_vec
    ↓  Hybrid Search (Qdrant)
top-40 candidatos (dense) + top-40 (sparse)
    ↓  Fusão RRF (k=60) + boost ementas x1.3
top-20 por score RRF
    ↓  Parent Lookup
top-20 contextos completos (texto_parent do payload)
    ↓  Reranking (bge-reranker-v2-m3)
top-5 contextos reordenados
    ↓  Geração (GPT-4o)
Resposta com citações [TIPO NUM/ANO, Art. X]
```

---

## Componentes

### Módulo 1 — Consolidação de Metadados
- **Input:** 3 JSONs brutos em `data/raw/`
- **Output:** `data/processed/metadata.parquet`
- Normalização de tipos (REN, REH, REA, DSP, PRT), datas, doc_ids canônicos
- Deduplicação por `doc_id + arquivo`
- Flag `revogada` baseada no campo `situacao`

### Módulo 2 — Download de PDFs
- Já executado externamente via Playwright
- 26.672 PDFs disponíveis (99.6% de cobertura)
- 96 PDFs inacessíveis (bloqueio no servidor ANEEL)

### Módulo 3 — Parsing de PDFs
- **Detecção de tipo:** native / scanned / mixed (threshold: 100 chars/página)
- **Texto nativo:** PyMuPDF com limpeza de hifenação e encoding
- **OCR:** pdf2image + Tesseract (lang=por+eng, psm=6)
- **Tabelas:** pdfplumber → Markdown pipe format
- **Limpeza:** remoção de cabeçalhos/rodapés repetidos (>50% páginas)
- **Incremental:** pula docs já processados com sucesso

### Módulo 4 — Chunking Document-Aware

Ver `docs/chunking_strategy.md` para detalhes completos.

Resultado: **228.078 chunks** distribuídos em:
- `child`: 111.531 (chunks indexáveis de artigos)
- `fallback`: 89.757 (docs sem estrutura de artigos)
- `ementa`: 26.768 (1 por documento, sempre)
- `table`: 22 (tabelas markdown)

### Módulo 5 — Embedding e Indexação
- **Modelo:** BAAI/bge-m3 (dense 1024-dim + sparse nativo)
- **Dispositivo:** CUDA (RTX 5070) — ~2-4h para 228k chunks
- **Qdrant local** (modo embarcado, SQLite)
- Indexados: chunks `child`, `ementa`, `fallback`, `table`
- `texto_parent` armazenado no payload (não indexado)

### Módulo 6 — Pipeline de Consulta

**Query Processing:**
- Extração de filtros via regex: ano (2016/2021/2022), tipo (REN/REH/...), número
- Limpeza de stop words jurídicas

**Hybrid Search:**
- Duas buscas paralelas (ThreadPoolExecutor): dense + sparse
- Filtros de payload: `ano`, `tipo_sigla`

**RRF (Reciprocal Rank Fusion):**
- `score_rrf(d) = Σ 1 / (k + rank(d)) ` com k=60
- Boost de 1.3x para chunks de ementa

**Parent Lookup:**
- Child → `texto_parent` do payload (artigo completo)
- Ementa/Fallback → próprio texto
- Deduplicação por `parent_id`

**Reranking:**
- BAAI/bge-reranker-v2-m3 (cross-encoder)
- `compute_score([(query, ctx)] * n, normalize=True)`
- Top-5 para o LLM

**Geração:**
- GPT-4o, temperatura 0.1, max 1500 tokens
- System prompt com instrução obrigatória de citação
- Retry automático (2x) em caso de falha de API

### Módulo 7 — Avaliação com RAGAS

**Golden Set:** 20 perguntas cobrindo:
- Definições em REN (microgeração, PRODIST, etc.)
- Perguntas de metadado direto
- Perguntas agregadas
- Perguntas sobre tabelas tarifárias (REH)

**Loop de Autocorreção:**
1. Geração normal (GPT-4o)
2. Juiz leve (GPT-4o-mini, 1 chamada): faithfulness score 0-1
3. Se score < 0.5: regenera com prompt mais restritivo
4. Máximo 3 tentativas → `needs_review` se ainda falhar

**Métricas RAGAS** (juiz: GPT-4o-mini):
- `faithfulness`: resposta sustentada no contexto?
- `answer_relevancy`: resposta relevante para a pergunta?
- `context_precision`: contextos recuperados são relevantes?
- `context_recall`: contextos cobrem a resposta esperada?

**Métricas de Retrieval:**
- Recall@5, Recall@10, Recall@20
- MRR (Mean Reciprocal Rank)
- Hit Rate@5

---

## Stack Técnica

| Componente | Ferramenta | Justificativa |
|---|---|---|
| Parsing PDF | PyMuPDF + Tesseract OCR | PyMuPDF para texto nativo; Tesseract para escaneados |
| Chunking | Python customizado | document-aware, controle total da hierarquia |
| Embedding | BAAI/bge-m3 | dense + sparse nativo, 8192 tokens, multilíngue |
| Vector DB | Qdrant (local embarcado) | hybrid search nativo, filtros por payload, sem infra |
| Reranker | BAAI/bge-reranker-v2-m3 | cross-encoder, maior ganho de qualidade pós-retrieval |
| LLM | GPT-4o | melhor em PT jurídico, instrução de citação |
| Juiz avaliação | GPT-4o-mini | mais barato para avaliação em batch |
| Avaliação | RAGAS | faithfulness, context precision/recall, answer relevance |

---

## Estrutura de Pastas

```
ANEEL-RAG/
├── CLAUDE.md                          # instruções do projeto
├── README.md                          # documentação principal
├── requirements.txt
├── .env                               # chaves de API (não versionado)
├── data/
│   ├── raw/                           # JSONs originais (não versionado)
│   ├── pdfs/                          # PDFs baixados (não versionado)
│   └── processed/
│       ├── metadata.parquet           # 26.768 documentos normalizados
│       ├── chunks.parquet             # 228.078 chunks
│       └── texts/                     # {doc_id}.txt por documento
├── src/
│   ├── config.py                      # configurações globais
│   ├── 01_consolidate_metadata.py
│   ├── 02_download_pdfs.py
│   ├── 03_parse.py
│   ├── 04_chunk.py
│   ├── 05_embed_index.py
│   ├── 06_query.py                    # pipeline de consulta
│   └── 07_evaluate.py                 # avaliação RAGAS
├── src/utils/
│   ├── token_counter.py
│   ├── text_utils.py
│   ├── qdrant_filters.py
│   ├── ids.py
│   ├── env_check.py
│   └── logger.py
├── tests/
│   ├── test_metadata.py
│   ├── test_parse.py
│   ├── test_chunk.py
│   ├── test_query.py
│   ├── test_evaluate.py
│   └── golden_set.json
├── notebooks/
│   ├── 01_explore_metadata.ipynb
│   ├── 02_inspect_chunks.ipynb
│   ├── 03_query_debug.ipynb
│   └── 04_eval_analysis.ipynb
├── docs/
│   ├── architecture.md                # este arquivo
│   ├── chunking_strategy.md
│   └── results.md
└── qdrant_db/                         # índice vetorial (não versionado)
```
