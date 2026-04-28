# ANEEL RAG — Sistema de Perguntas sobre Legislação do Setor Elétrico

Sistema de Retrieval-Augmented Generation (RAG) para responder perguntas sobre legislação do setor elétrico brasileiro.  
Dataset: documentos reais da ANEEL dos anos 2016, 2021 e 2022 — ~26.768 documentos, ~259.187 chunks indexados.

---

## Início Rápido

> Siga os passos abaixo em ordem. O sistema estará rodando em menos de 10 minutos.

### Passo 1 — Clonar o repositório

```bash
git clone https://github.com/murilohonorato/ANEEL-RAG.git
cd ANEEL-RAG
```

### Passo 2 — Instalar dependências

```bash
pip install -r requirements.txt
```

### Passo 3 — Configurar a chave de API

Crie um arquivo `.env` na **raiz do projeto** (`ANEEL-RAG/.env`):

```env
OPENAI_API_KEY=sk-...
QDRANT_PATH=qdrant_db/
DATA_DIR=data/
```

> A chave OpenAI é necessária apenas para geração de respostas (GPT-4o). Embedding e reranking rodam localmente.

### Passo 4 — Baixar o banco vetorial

O índice vetorial (~3 GB) não está no repositório. Baixe pelo Google Drive:

**[Download qdrant_db — Google Drive](https://drive.google.com/drive/folders/1VJl3-fqbafIdq9ANr4CSanXvhRsztHpi?usp=drive_link)**

#### Opção A — Download manual (recomendado)

1. Acesse o link acima
2. Clique com o botão direito na pasta `qdrant_db` → **"Fazer download"**
3. O Google Drive vai zipar automaticamente — aguarde
4. Mova e descompacte na raiz do projeto:

```powershell
# Windows (PowerShell) — execute dentro da pasta ANEEL-RAG/
Expand-Archive "$env:USERPROFILE\Downloads\qdrant_db.zip" .
```

```bash
# Linux / Mac — execute dentro da pasta ANEEL-RAG/
unzip ~/Downloads/qdrant_db.zip
```

#### Opção B — Download via Python (gdown)

```bash
pip install gdown
gdown --folder 1VJl3-fqbafIdq9ANr4CSanXvhRsztHpi --remaining-ok
```

> O `gdown` cria a pasta `qdrant_db/` automaticamente no diretório atual.

#### Onde colocar a pasta

A estrutura final deve ser **exatamente assim**:

```
ANEEL-RAG/          ← raiz do projeto (onde está o README.md)
├── qdrant_db/      ← pasta baixada do Drive
│   └── collection/
│       └── aneel_legislacao/
├── src/
├── .env
└── ...
```

#### Verificar se está correto

```bash
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(path='qdrant_db')
info = client.get_collection('aneel_legislacao')
print(f'OK — {info.points_count:,} pontos indexados')
# Esperado: 259.187
"
```

### Passo 5 — Rodar a interface

```bash
streamlit run src/app.py
```

Acesse **http://localhost:8501** no navegador. Na primeira execução os modelos (~3 GB) são baixados automaticamente.

---

## Alternativa: Docker

Se preferir rodar sem instalar nada localmente (requer Docker instalado):

```bash
# 1. Certifique-se que a pasta qdrant_db/ está na raiz (Passo 4 acima)
# 2. Crie o .env com sua chave (Passo 3 acima)
# 3. Suba o container
docker-compose up --build

# Acesse http://localhost:8501
```

> Na primeira execução os modelos são baixados e cacheados em volume Docker.

---

## Requisitos

| Requisito | Obrigatório | Observação |
|-----------|------------|------------|
| Python 3.11+ | ✅ | |
| Chave OpenAI | ✅ | Para geração com GPT-4o |
| CUDA / GPU | Recomendado | Embedding e reranking ~10× mais rápido |
| Tesseract OCR | Só para re-parsear PDFs | Não necessário para consultas |
| Docker | Apenas para opção Docker | |

---

## Modo Linha de Comando

```bash
# Pergunta única
python src/06_query.py --question "O que é microgeração distribuída?"

# Modo interativo
python src/06_query.py

# Com debug (mostra chunks recuperados e scores)
python src/06_query.py --question "Qual a situação da REN 1000/2021?" --debug
```

---

## Arquitetura

```
[FASE OFFLINE — executada uma vez]
JSON bruto → metadados → PDFs → texto limpo → chunks → Qdrant

[FASE ONLINE — por consulta]
pergunta → query processing → embedding → hybrid search → RRF → reranking → GPT-4o → resposta
```

Ver `docs/architecture.md` para diagrama completo e `docs/chunking_strategy.md` para detalhes do chunking.

---

## Rodando o Pipeline Completo (opcional)

> Só necessário se quiser re-indexar tudo do zero. O banco vetorial do Drive já está pronto.

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
# Output: data/processed/chunks.parquet
```

### Passo 4 — Embedding e indexação

```bash
python src/05_embed_index.py
# Requer GPU — ~2h em RTX 5070
# Output: qdrant_db/
```

> ⚠️ Este passo exige GPU. Em CPU pode levar 60+ horas.

### Passo 5 — Avaliar com RAGAS

```bash
python src/07_evaluate.py
# Output: results/eval_report_{timestamp}.md
# Golden set: tests/golden_set.json (20 perguntas anotadas)
# Custo estimado: ~$0.25 em API OpenAI
```

---

## Testes

```bash
pytest tests/ -v
```

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Precisão — texto corrido | ~80% |
| Precisão — tabelas | ~0% |
| Respostas com alucinação | ~10% |
| Latência média por query | ~5s |

Ver `docs/results.md` para análise completa.

---

## Stack Técnica

| Componente | Ferramenta |
|-----------|-----------|
| Interface | Streamlit + Docker |
| Parsing PDF | PyMuPDF + Tesseract OCR + pdfplumber |
| Chunking | Python customizado (document-aware) |
| Embedding | BAAI/bge-m3 (dense 1024-dim + sparse BM25) |
| Vector DB | Qdrant (local embarcado) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | GPT-4o (temperatura 0.1) |
| Avaliação | RAGAS + GPT-4o-mini |

---

## Estrutura de Pastas

```
ANEEL-RAG/
├── README.md
├── requirements.txt
├── .env                         # chaves de API (não versionado)
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── app.py                   # interface Streamlit
│   ├── config.py
│   ├── 01_consolidate_metadata.py
│   ├── 02_download_pdfs.py      # já executado ✅
│   ├── 03_parse.py
│   ├── 04_chunk.py
│   ├── 05_embed_index.py
│   ├── 06_query.py              # pipeline de consulta
│   └── 07_evaluate.py
├── src/utils/
├── tests/
│   └── golden_set.json          # 20 perguntas anotadas
├── notebooks/
├── docs/
│   ├── architecture.md
│   ├── chunking_strategy.md
│   └── results.md
└── qdrant_db/                   # índice vetorial — baixar do Drive (não versionado)
```

---

## Documentação

- `docs/architecture.md` — Arquitetura completa com diagrama do pipeline
- `docs/chunking_strategy.md` — Estratégia Parent-Child Document-Aware
- `docs/results.md` — Métricas de avaliação e análise qualitativa

---

## Grupo

Projeto desenvolvido para a disciplina de NLP/RAG aplicado à regulação do setor elétrico brasileiro.

**Autores:**
- Murilo Honorato de Souza
- Lucas Boclin Cunha Borges
- Lucas Honorato de Souza
