# ANEEL RAG — Contexto do Projeto

## Objetivo
Sistema RAG para responder perguntas sobre legislação do setor elétrico brasileiro.
Dataset: documentos reais da ANEEL (2016, 2021, 2022).
Avaliação: benchmark de perguntas anotado por especialista.
**Foco principal: qualidade do pipeline de recuperação, não o agente.**

---

## Setup — Primeiros Passos

### 1. Clonar e instalar dependências
```bash
git clone https://github.com/murilohonorato/ANEEL-RAG.git
cd ANEEL-RAG
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente
Crie um arquivo `.env` na raiz (não versionar — já está no `.gitignore`):
```
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_PATH=qdrant_db/
DATA_DIR=data/
```

### 3. Colocar os arquivos de dados
Os JSONs e PDFs **não estão no repositório** (arquivos grandes). Receba-os com o grupo e coloque-os assim:

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

### 4. Status atual dos dados
- **Todos os PDFs já foram baixados** (2016, 2021, 2022) ✅
- O Módulo 2 (download) está concluído — não é necessário rodar `02_download_pdfs.py`
- O próximo passo de implementação é o **Módulo 1** (`src/01_consolidate_metadata.py`)

### 5. Verificar instalação
```bash
python -c "import fitz, qdrant_client, FlagEmbedding, anthropic; print('OK')"
pytest tests/ -v
```

---

## Dataset
- 3 arquivos JSON em `data/raw/`: metadados de legislação ANEEL por ano (nomes exatos acima)
- Estrutura do JSON: `{data_key: {status, registros: [{titulo, autor, ementa, assunto, situacao, pdfs: [{url, arquivo, baixado}]}]}}`
- ~18.688 registros únicos, ~27.039 referências a PDFs
- PDFs armazenados em `data/pdfs/{ano}/` — **todos já baixados** ✅
- Tipos de documento: DSP (despacho ~53%), REA (21%), PRT (17%), REH (2.5%), REN (0.8%)
- **REN e REH são os mais relevantes para o benchmark** — normas de maior peso regulatório

## Estratégia de Chunking — Parent-Child Document-Aware

### Três tipos de chunk por documento:

**1. Chunk de Ementa** (prioridade alta no retrieval)
- Gerado direto do JSON, sem precisar do PDF
- Contém: tipo, número, ano, autor, assunto, ementa completa, situação, data publicação
- 1 por documento
- Recebe score boost no Qdrant

**2. Chunk Parent** (contexto completo para o LLM)
- Artigo completo + todos os seus parágrafos/incisos
- 400–800 tokens
- NÃO é indexado para busca vetorial
- Armazenado no payload do Qdrant (campo `texto_parent`)
- Recuperado via `parent_id` quando um child é encontrado

**3. Chunk Child** (indexado para busca)
- Parágrafo ou grupo de incisos (subdivisão do artigo)
- 100–200 tokens
- Embedado e indexado no Qdrant
- Carrega `parent_id` no payload para recuperar o parent

### Lógica de decisão:
```
texto do PDF
    ↓
detecta "Art. Xº" via regex?
    ├── SIM → chunk por artigo (parent) → divide em parágrafos (children)
    └── NÃO → RecursiveCharacterSplitter(size=500, overlap=80) como parent/child
         ↓
sempre criar chunk de ementa separado (do JSON)
```

### Prefixo contextual em cada chunk:
Todo chunk começa com: `{TIPO} {NUMERO}/{ANO} — {primeiros 150 chars da ementa}`
Garante que o embedding carrega a identidade do documento.

## Stack Técnica

| Componente | Ferramenta | Motivo |
|---|---|---|
| Parsing PDF | PyMuPDF (fitz) + Tesseract OCR | PyMuPDF para texto nativo; Tesseract para escaneados |
| Chunking | custom Python (regex hierárquico) | document-aware, controle total |
| Embedding | BAAI/bge-m3 | dense + sparse nativo, 8192 tokens, multilíngue, open |
| Vector DB | Qdrant (local, modo embarcado) | hybrid search nativo, filtros por payload, sem infra |
| Reranker | BAAI/bge-reranker-v2-m3 | cross-encoder, maior ganho de qualidade pós-retrieval |
| LLM | Claude Sonnet (API Anthropic) | melhor em PT jurídico, instrução de citação obrigatória |
| Avaliação | RAGAS | faithfulness, context precision/recall, answer relevance |

## Pipeline Online (por consulta)

```
pergunta do usuário
    ↓
1. QUERY PROCESSING — extrai filtros (ano, tipo, número) + reescreve query
    ↓
2. EMBEDDING — bge-m3 gera dense_vec + sparse_vec da query
    ↓
3. HYBRID SEARCH — Qdrant busca children + ementas (dense + sparse + filtros)
    ↓
4. FUSÃO RRF — combina rankings, ementas com boost
    ↓ top-20 children/ementas
5. PARENT LOOKUP — para cada child: recupera texto_parent do payload
    ↓ top-20 parents completos
6. RERANKING — bge-reranker-v2-m3 reordena com cross-encoder
    ↓ top-5 parents
7. GERAÇÃO — Claude Sonnet com instrução de citação [TIPO NUMERO/ANO, Art. X]
    ↓
resposta com citações
```

## Estrutura de Pastas

```
ANEEL-RAG/
├── CLAUDE.md                  ← este arquivo
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                   ← JSONs originais (não versionar)
│   ├── pdfs/                  ← PDFs baixados (não versionar)
│   │   ├── 2016/
│   │   ├── 2021/
│   │   └── 2022/
│   └── processed/             ← parquets intermediários
│       ├── metadata.parquet   ← metadados consolidados
│       ├── chunks.parquet     ← todos os chunks com metadados
│       └── index_stats.json   ← estatísticas do índice
├── src/
│   ├── 01_consolidate_metadata.py   ← JSON → metadata.parquet
│   ├── 02_download_pdfs.py          ← download com undetected_chromedriver (já executado ✅)
│   ├── 03_parse.py                  ← PDF → texto limpo
│   ├── 04_chunk.py                  ← texto → chunks parent/child
│   ├── 05_embed_index.py            ← chunks → Qdrant
│   ├── 06_query.py                  ← pipeline de consulta completo
│   └── 07_evaluate.py               ← golden set + RAGAS
├── tests/
│   ├── test_chunk.py
│   ├── test_parse.py
│   └── test_query.py
├── notebooks/
│   └── exploration.ipynb
└── qdrant_db/                 ← índice vetorial (não versionar)
```

## Metadados de Cada Chunk (payload Qdrant)

```python
{
    # Identidade do documento
    "doc_id":        "ren20211000ti",      # nome do arquivo PDF
    "tipo_sigla":    "REN",
    "tipo_nome":     "Resolução Normativa",
    "numero":        "1000",
    "ano":           2021,
    "data_pub":      "2021-12-07",
    "data_ass":      "2021-12-01",
    "autor":         "ANEEL",
    "assunto":       "Distribuição de energia elétrica",
    "ementa":        "Estabelece os Procedimentos...",
    "situacao":      "NÃO CONSTA REVOGAÇÃO EXPRESSA",
    "pdf_url":       "http://www2.aneel.gov.br/cedoc/ren20211000ti.pdf",

    # Identidade do chunk
    "chunk_type":    "child",   # "child" | "ementa" | "fallback"
    "parent_id":     "ren20211000ti_art003",
    "texto_parent":  "Art. 3º Para os fins desta Resolução...",  # texto completo do parent
    "art_num":       3,

    # Texto do chunk (child)
    "texto":         "REN 1000/2021 — Art. 3º\n§ 1º Considera-se...",
}
```

## Roadmap e Progresso

O arquivo `roadmap.md` na raiz do projeto contém todos os módulos de implementação com checkboxes.

**REGRA OBRIGATÓRIA:** toda vez que um módulo (ou sub-tarefa de um módulo) for implementado e testado, o checkbox correspondente no `roadmap.md` **deve ser marcado** imediatamente:
- Sub-tarefa concluída: `- [ ]` → `- [x]`
- Módulo inteiro concluído: marcar todas as sub-tarefas E atualizar a tabela de "Status Geral" no final do roadmap

Não considere nenhum módulo finalizado sem testes passando (`pytest tests/`).

## Convenções de Código

- Python 3.11+
- Type hints em todas as funções
- Logging via `logging` (não print)
- Cada script é executável standalone: `python src/04_chunk.py`
- Parâmetros configuráveis no topo de cada arquivo na seção `# CONFIG`
- Salvar artefatos intermediários em `data/processed/` para não reprocessar
- Testes com pytest: `pytest tests/`

## Observações Importantes

1. **OCR**: PDFs antigos da ANEEL frequentemente são escaneados. Detectar via densidade de texto/página < 100 chars → rodar Tesseract.
2. **Deduplicação**: usar nome do arquivo (`arquivo` no JSON) como ID único, não a URL.
3. **Revogação**: campo `situacao` indica se a norma foi revogada. Filtrar/rankear por isso.
4. **Número da norma**: normalizar variações ("REN 1000", "REN 1.000", "Resolução 1000/2021") antes de indexar e buscar.
5. **Ementas são ouro**: texto escrito por especialista, ideal para retrieval de perguntas gerais.
6. **Despachos (DSP)**: 53% do volume, geralmente curtos, sem estrutura de artigos — usar fallback recursivo.
