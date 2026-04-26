# ANEEL RAG — Sistema de Perguntas sobre Legislação do Setor Elétrico

Sistema de Retrieval-Augmented Generation (RAG) para responder perguntas sobre legislação do setor elétrico brasileiro. Dataset: documentos reais da ANEEL dos anos 2016, 2021 e 2022 (~26.768 documentos, ~259.187 chunks indexados).

---

## Arquitetura

```
[FASE OFFLINE — indexação]
JSON bruto → metadados → PDFs → texto limpo → chunks → Qdrant

[FASE ONLINE — por consulta]
pergunta → query processing → embedding → hybrid search → RRF → reranking → GPT-4o → resposta
```

Ver `docs/architecture.md` para diagrama completo e `docs/chunking_strategy.md` para detalhes do chunking.

---

## Interface Gráfica (recomendado)

A forma mais simples de usar o sistema é pela interface web:

### Opção 1 — Docker (sem instalar nada)

```bash
# 1. Obter a pasta qdrant_db/ com o grupo e colocá-la na raiz do projeto
# 2. Criar o arquivo .env
echo OPENAI_API_KEY=sk-... > .env

# 3. Subir o container
docker-compose up --build

# 4. Acessar no navegador
# http://localhost:8501
```

> Na primeira execução os modelos (~3GB) são baixados automaticamente e cacheados para as próximas.

### Opção 2 — Local

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

---

## Requisitos

- Python 3.11+
- CUDA (recomendado para embedding e reranking — RTX ou melhor)
- Tesseract OCR instalado no sistema
- Chave de API OpenAI (para geração com GPT-4o)

---

## Instalação

```bash
git clone https://github.com/murilohonorato/ANEEL-RAG.git
cd ANEEL-RAG
pip install -r requirements.txt
```

### Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sk-...
QDRANT_PATH=qdrant_db/
DATA_DIR=data/
```

### Verificar instalação

```bash
python -c "import fitz, qdrant_client, FlagEmbedding, openai; print('OK')"
pytest tests/ -v
```

---

## Dados

Os arquivos de dados não estão no repositório (muito grandes). Estrutura esperada:

```
data/
├── raw/
│   ├── biblioteca_aneel_gov_br_legislacao_2016_metadados.json
│   ├── biblioteca_aneel_gov_br_legislacao_2021_metadados.json
│   └── biblioteca_aneel_gov_br_legislacao_2022_metadados.json
└── pdfs/
    ├── 2016/    ← PDFs do ano 2016
    ├── 2021/    ← PDFs do ano 2021
    └── 2022/    ← PDFs do ano 2022
```

**Status atual:**
- PDFs já baixados (Módulo 2 concluído externamente) ✅
- Embedding já gerado e indexado no Qdrant ✅ (`qdrant_db/` — obter com o grupo)

---

## Rodando o Pipeline Completo

### Passo 1 — Consolidar metadados

```bash
python src/01_consolidate_metadata.py
# Output: data/processed/metadata.parquet (26.768 documentos)
```

### Passo 2 — Parsear PDFs

```bash
python src/03_parse.py
# Output: data/processed/texts/{doc_id}.txt
# Usa OCR automaticamente para PDFs escaneados
```

### Passo 3 — Gerar chunks

```bash
python src/04_chunk.py
# Output: data/processed/chunks.parquet (228.078 chunks)
```

### Passo 4 — Embedding e indexação

```bash
python src/05_embed_index.py
# Requer GPU (recomendado) — ~2h em RTX 5070
# Output: qdrant_db/ (índice vetorial local)
```

> ⚠️ Este passo exige GPU. Em CPU pode levar 60+ horas. Use Google Colab com GPU ou máquina do grupo.

### Passo 5 — Consultar

```bash
# Modo interativo
python src/06_query.py

# Modo single query
python src/06_query.py --question "O que é microgeração distribuída?"

# Com debug (mostra chunks recuperados e scores)
python src/06_query.py --question "Qual a situação da REN 1000/2021?" --debug
```

### Passo 6 — Avaliar com RAGAS

```bash
python src/07_evaluate.py
# Output: results/eval_report_{timestamp}.md
# Usa golden set de 20 perguntas anotadas (tests/golden_set.json)
# Custo estimado: ~$0.25 em API OpenAI
```

---

## Testes

```bash
# Todos os testes
pytest tests/ -v

# Por módulo
pytest tests/test_metadata.py -v
pytest tests/test_parse.py -v
pytest tests/test_chunk.py -v
pytest tests/test_query.py -v
pytest tests/test_evaluate.py -v
```

---

## Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Conteúdo |
|----------|----------|
| `01_explore_metadata.ipynb` | Distribuição de tipos, anos, ementas |
| `02_inspect_chunks.ipynb` | Exemplos de chunks, distribuição de tokens |
| `03_query_debug.ipynb` | Debug passo a passo de queries |
| `04_eval_analysis.ipynb` | Análise dos resultados de avaliação |

---

## Resultados

Ver `docs/results.md` para métricas completas. Estimativas:

| Métrica | Valor |
|---------|-------|
| Recall@5 | ~0.75 |
| Recall@10 | ~0.85 |
| MRR | ~0.70 |
| RAGAS faithfulness | ~0.80 |
| RAGAS answer_relevancy | ~0.85 |

---

## Stack Técnica

| Componente | Ferramenta |
|-----------|-----------|
| Interface | Streamlit + Docker |
| Parsing PDF | PyMuPDF + Tesseract OCR |
| Chunking | Python customizado (document-aware) |
| Embedding | BAAI/bge-m3 (dense 1024-dim + sparse) |
| Vector DB | Qdrant (local embarcado) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | GPT-4o (temperatura 0.1) |
| Juiz avaliação | GPT-4o-mini |
| Avaliação | RAGAS |

---

## Estrutura de Pastas

```
ANEEL-RAG/
├── CLAUDE.md                    # instruções do projeto (para Claude Code)
├── README.md                    # este arquivo
├── requirements.txt
├── .env                         # chaves de API (não versionado)
├── data/
│   ├── raw/                     # JSONs originais (não versionado)
│   ├── pdfs/                    # PDFs baixados (não versionado)
│   └── processed/
│       ├── metadata.parquet     # 26.768 documentos normalizados
│       ├── chunks.parquet       # 228.078 chunks
│       └── texts/               # {doc_id}.txt por documento
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── app.py                   # interface Streamlit
│   ├── config.py                # configurações globais
│   ├── 01_consolidate_metadata.py
│   ├── 02_download_pdfs.py      # já executado ✅
│   ├── 03_parse.py
│   ├── 04_chunk.py
│   ├── 05_embed_index.py
│   ├── 06_query.py              # pipeline de consulta
│   └── 07_evaluate.py           # avaliação RAGAS
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
│   └── golden_set.json          # 20 perguntas anotadas
├── notebooks/
│   ├── 01_explore_metadata.ipynb
│   ├── 02_inspect_chunks.ipynb
│   ├── 03_query_debug.ipynb
│   └── 04_eval_analysis.ipynb
├── docs/
│   ├── architecture.md          # arquitetura completa do sistema
│   ├── chunking_strategy.md     # estratégia parent-child document-aware
│   └── results.md               # métricas e análise de resultados
└── qdrant_db/                   # índice vetorial (não versionado)
```

---

## Documentação

- `docs/architecture.md` — Arquitetura completa com diagrama do pipeline
- `docs/chunking_strategy.md` — Estratégia Parent-Child Document-Aware com exemplos reais
- `docs/results.md` — Métricas de avaliação e análise qualitativa

---

## Grupo

Projeto desenvolvido para o grupo de estudos de NLP/RAG aplicado à regulação do setor elétrico brasileiro.
