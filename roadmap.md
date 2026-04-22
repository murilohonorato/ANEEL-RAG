# ANEEL RAG — Roadmap de Implementação

> **Convenção:** marque `[x]` em cada módulo **e** em cada sub-tarefa assim que estiver implementado e testado.  
> Cada módulo tem um script correspondente em `src/`. Nenhum módulo deve ser considerado concluído sem testes passando.

---

## Visão Geral do Pipeline

```
[FASE OFFLINE — indexação]
JSON bruto → consolidar metadados → baixar PDFs → parsear PDFs → chunking
→ embedding (bge-m3) → indexar no Qdrant

[FASE ONLINE — consulta]
pergunta → query processing → embedding da query → hybrid search (dense+sparse)
→ fusão RRF → parent lookup → reranking (bge-reranker-v2-m3) → geração (Claude Sonnet)
→ resposta com citações

[AVALIAÇÃO]
golden set → RAGAS → métricas (faithfulness, context precision/recall, answer relevance)
```

---

## Módulo 0 — Infraestrutura e Ambiente

- [x] **0.1 — Dependências**
  - Arquivo `requirements.txt` consolidado com todas as bibliotecas e versões fixas
  - Dependências: `pymupdf>=1.24`, `pytesseract`, `ocrmypdf`, `pandas>=2.0`, `pyarrow`, `tiktoken`, `FlagEmbedding>=1.2`, `fastembed`, `qdrant-client>=1.9`, `anthropic>=0.25`, `ragas>=0.1`, `tqdm`, `loguru`, `pytest`
  - Instruções de instalação no README: `pip install -r requirements.txt`

- [x] **0.2 — Variáveis de ambiente**
  - Arquivo `.env.example` com todas as chaves necessárias: `ANTHROPIC_API_KEY`, `QDRANT_PATH`, `DATA_DIR`
  - `.env` no `.gitignore` (já configurado)
  - Módulo `src/config.py` que carrega `.env` via `python-dotenv` e expõe constantes globais

- [x] **0.3 — Estrutura de pastas**
  - Garantir existência de: `data/raw/`, `data/pdfs/2016/`, `data/pdfs/2021/`, `data/pdfs/2022/`, `data/processed/`, `qdrant_db/`, `results/`, `logs/`
  - Script utilitário `src/setup_dirs.py` que cria toda a estrutura
  - Verificar que `.gitignore` exclui `data/pdfs/`, `qdrant_db/`, `data/processed/`, `*.log`, `*.pdf`

- [x] **0.4 — Logger global**
  - Configurar `loguru` como logger padrão em `src/utils/logger.py`
  - Formato: `[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {module}:{line} — {message}`
  - Rotação de logs: 10 MB por arquivo, retenção de 7 dias
  - Sink para arquivo `logs/pipeline.log` + console colorido

- [x] **0.5 — Testes de sanidade do ambiente**
  - `tests/test_environment.py`: verifica que todas as importações funcionam, GPU/MPS detectada, Qdrant abre em modo embarcado
  - Checklist de hardware: detectar se está em Mac M4 (MPS), Linux com CUDA, ou CPU fallback
  - ⚠️ `test_python_version` falha pois a máquina tem Python 3.9 — instalar 3.11+ via pyenv antes de rodar o pipeline completo

---

## Módulo 1 — Consolidação de Metadados (`src/01_consolidate_metadata.py`)

**Objetivo:** Ler os 3 JSONs brutos (2016, 2021, 2022) e produzir um DataFrame limpo e normalizado em `data/processed/metadata.parquet`.

- [x] **1.1 — Leitura e parsing dos JSONs**
  - Carregar os 3 arquivos em `data/raw/`: `biblioteca_aneel_gov_br_legislacao_2016_metadados.json`, `*_2021_*`, `*_2022_*`
  - Estrutura esperada: `{data_key: {status, registros: [{titulo, autor, ementa, assunto, situacao, pdfs: [{url, arquivo, baixado}]}]}}`
  - Função `parse_year_json(path: Path, year: int) -> list[dict]` que itera sobre todas as datas e registros
  - Extrair: `titulo`, `autor`, `ementa`, `assunto`, `situacao`, `data` (chave do JSON), lista de PDFs

- [x] **1.2 — Normalização de tipos de documento**
  - Extrair `tipo_sigla` do nome do arquivo PDF: regex `r'^([a-zA-Z]+)(\d{4})(\d+)'` sobre o campo `arquivo`
  - Mapeamento canônico de siglas:
    - `ren` → `REN` → "Resolução Normativa"
    - `reh` → `REH` → "Resolução Homologatória"
    - `rea` → `REA` → "Resolução Autorizativa"
    - `dsp` → `DSP` → "Despacho"
    - `prt` → `PRT` → "Portaria"
    - outros → `OUTRO`
  - Extrair `numero` (int) e `ano` do arquivo para validação cruzada com o JSON

- [x] **1.3 — Normalização de número da norma**
  - Campo `numero_norm` como string zero-padded: `f"{numero:04d}"`
  - Campo `doc_id` canônico: `f"{tipo_sigla.lower()}{ano}{numero_norm}"` (ex: `ren20211000`)
  - Tratar variações nos nomes: remover acentos, converter para minúsculo, strip whitespace

- [x] **1.4 — Normalização de datas**
  - Campo `data_pub`: parse de strings no formato `DD/MM/YYYY`, `YYYY-MM-DD`, ou extrair do campo `data` da chave JSON
  - Converter para `datetime.date` e serializar como ISO string `YYYY-MM-DD`
  - Campos: `data_pub` (data da publicação no DOU), `ano` (int)

- [x] **1.5 — Deduplicação**
  - Usar `doc_id` + `arquivo` como chave primária
  - Registros com mesmo `arquivo` em anos diferentes → manter o mais recente
  - Logar quantidade de duplicatas removidas por ano

- [x] **1.6 — Flag de situação**
  - Campo `revogada` (bool): `True` se `situacao` contém palavras como "revogad", "cancelad", "substituíd"
  - Manter campo `situacao` original também

- [x] **1.7 — Enriquecimento de URLs dos PDFs**
  - Para cada PDF: validar que URL começa com `http` e termina em `.pdf`
  - Campo `pdf_filename` = `f"{doc_id}.pdf"` (nome canônico para disco)
  - Campo `pdf_path_expected` = `f"data/pdfs/{ano}/{pdf_filename}"`
  - Campo `pdf_baixado` (bool): verificar se arquivo existe no disco

- [x] **1.8 — Salvar metadados**
  - Output: `data/processed/metadata.parquet` com todos os campos normalizados
  - Output secundário: `data/processed/metadata.csv` para inspeção manual
  - Output de stats: `data/processed/index_stats.json`
  - Log completo de anomalias: registros sem PDF, tipo desconhecido, data inválida

- [x] **1.9 — Testes do módulo**
  - `tests/test_metadata.py`: 18 testes, todos passando
    - `test_no_duplicate_doc_ids`, `test_tipo_sigla_known_values`, `test_data_pub_format`, `test_doc_id_regex`
    - + testes unitários de `strip_label`, `normalize_date`, `extract_tipo_info`, `is_revogada`, `parse_year_json`
  - ⚠️ Dados reais têm padrão `sn` (sem número) em ~91 arquivos — mapeados como OUTRO corretamente

---

## Módulo 2 — Download de PDFs (`src/02_download_pdfs.py`)

**Objetivo:** Baixar todos os PDFs referenciados nos metadados, respeitando os checkpoints de progresso já existentes (script `extract_pdfs/` já funcional para 2021).

- [x] **2.1 — Unificação dos scripts de download**
- [x] **2.2 — Checkpoint e retomada**
- [x] **2.3 — Driver undetected_chromedriver**
- [x] **2.4 — Download com espera por arquivo**
- [x] **2.5 — Relatório de erros**

> ✅ **Concluído externamente** — todos os PDFs (2016, 2021, 2022) foram baixados via `extract_pdfs/` e colocados diretamente em `data/pdfs/{ano}/`. Os 3 JSONs estão em `data/raw/`. Não é necessário rodar `02_download_pdfs.py`.

---

## Módulo 3 — Parsing de PDFs (`src/03_parse.py`)

**Objetivo:** Extrair texto limpo de cada PDF, detectar documentos escaneados e aplicar OCR, converter tabelas para Markdown. Output: `data/processed/parsed_docs.parquet`.

- [x] **3.1 — Detecção de tipo de PDF**
  - Função `detect_pdf_type(path: Path) -> Literal["native", "scanned", "mixed"]`
  - Critério: abrir com `fitz` (PyMuPDF), contar caracteres por página
  - `native`: média > 100 chars/página
  - `scanned`: média < 100 chars/página  
  - `mixed`: algumas páginas nativas, outras escaneadas (threshold por página)
  - Logar distribuição de tipos ao final do processamento

- [x] **3.2 — Extração de texto nativo (PyMuPDF)**
  - Função `extract_text_native(path: Path) -> list[dict]`
  - Para cada página: `page.get_text("text")` com flags para preservar quebras de parágrafo
  - Resultado: lista de `{"page": N, "text": "...", "char_count": N}`
  - Limpeza básica: remover caracteres de controle, normalizar espaços múltiplos, normalizar hifenação no fim de linha (`word-\nword` → `wordword`)

- [x] **3.3 — OCR com Tesseract (PDFs escaneados)**
  - Função `extract_text_ocr(path: Path) -> list[dict]`
  - Usar `ocrmypdf` para gerar PDF com camada de texto: `ocrmypdf --language por --optimize 0 input.pdf output_ocr.pdf`
  - Extrair texto do PDF gerado com PyMuPDF
  - Alternativa rápida: `pdf2image` → `pytesseract.image_to_string(img, lang="por")`
  - Configuração Tesseract: `--psm 6` (bloco de texto uniforme), `lang=por+eng`

- [x] **3.4 — Extração de tabelas (pdfplumber)**
  - Função `extract_tables_as_markdown(path: Path) -> list[dict]`
  - Para cada página: `pdfplumber.open(path).pages[i].extract_tables()`
  - Converter cada tabela para Markdown usando função `table_to_markdown(table: list[list]) -> str`
  - Formato de output: `| col1 | col2 | col3 |\n|---|---|---|\n| val1 | val2 | val3 |`
  - Substituir a região da tabela no texto extraído pela versão Markdown
  - Prioridade: REH (resoluções homologatórias) contêm tabelas tarifárias críticas

- [x] **3.5 — Detecção de estrutura legal**
  - Função `detect_legal_structure(text: str) -> dict`
  - Detectar: presença de artigos (`Art. \d+`), parágrafos (`§ \d+`), incisos (`[IVX]+\s*—`), alíneas (`[a-z]\)`), anexos (`ANEXO`)
  - Retornar: `{"has_articles": bool, "article_count": int, "has_paragraphs": bool, ...}`
  - Usado pelo Módulo 4 para escolher estratégia de chunking

- [x] **3.6 — Limpeza de texto final**
  - Remover cabeçalhos e rodapés repetitivos (ex: "Diário Oficial da União", número de página)
  - Detectar linhas que aparecem em >50% das páginas → remover
  - Normalizar encoding: converter para UTF-8 puro, remover BOM
  - Normalizar aspas curvas para retas, travessões para ` — `
  - Remover `\x00` e outros caracteres nulos

- [x] **3.7 — Salvar docs parseados**
  - Output: `data/processed/parsed_docs.parquet` com colunas:
    - `doc_id`, `tipo_sigla`, `numero`, `ano`, `full_text`, `page_count`, `char_count`, `pdf_type`, `has_tables`, `has_articles`, `article_count`, `parse_method` ("native" | "ocr" | "mixed"), `parse_timestamp`
  - Sem limite de texto no parquet (usar `pyarrow` com `large_string` para textos longos)
  - Log de anomalias: PDFs vazios, falha de OCR, encoding incorreto

- [x] **3.8 — Testes do módulo**
  - `tests/test_parse.py`:
    - `test_native_pdf_non_empty`: PDF nativo retorna texto não vazio
    - `test_ocr_pdf_detected`: PDF escaneado detectado corretamente
    - `test_table_markdown_format`: tabelas convertidas com `|` correto
    - `test_encoding_clean`: sem `\x00` nem caracteres de controle no output
    - `test_legal_structure_ren`: REN com artigos detecta `has_articles=True`

---

## Módulo 4 — Chunking Document-Aware (`src/04_chunk.py`)

**Objetivo:** Transformar cada documento parseado em três tipos de chunk (ementa, parent, child) seguindo a estratégia document-aware com prefixo contextual.

- [x] **4.1 — Chunk de Ementa (do JSON)**
  - Função `build_ementa_chunk(row: pd.Series) -> dict`
  - Input: linha do `metadata.parquet`
  - Texto do chunk:
    ```
    {TIPO} {NUMERO}/{ANO} — {ementa}
    
    Tipo: {tipo_nome}
    Número: {numero}
    Ano: {ano}
    Autor: {autor}
    Assunto: {assunto}
    Situação: {situacao}
    Data de publicação: {data_pub}
    ```
  - Campos do payload: todos os metadados + `chunk_type="ementa"`, `parent_id=None`, `texto_parent=None`
  - Gerado para TODOS os documentos, independente de ter PDF baixado
  - Score boost no Qdrant: campo `is_ementa=True` para filtro/boost na query

- [x] **4.2 — Detecção de artigos via regex**
  - Função `split_by_articles(text: str) -> list[dict]`
  - Regex principal: `r'(?:^|\n)\s*(Art\.?\s*\d+[°º]?\.?\s*[-—]?\s*)'` (multiline)
  - Capturar: número do artigo, texto completo até o próximo artigo
  - Tratar casos especiais: "Art. 1º-A", "ARTIGO 5", "Art. 10"
  - Output: `[{"art_num": 1, "art_text": "Art. 1º ..."}, ...]`
  - Se `article_count == 0`: usar fallback recursivo (4.4)

- [x] **4.3 — Chunks Parent (por artigo)**
  - Função `build_parent_chunk(doc_row, art_num: int, art_text: str) -> dict`
  - Texto: prefixo contextual + texto completo do artigo + parágrafos/incisos
  - Tamanho: 400–800 tokens (usar `tiktoken` com modelo `cl100k_base` para contagem)
  - Se artigo > 800 tokens: dividir em múltiplos parents preservando `§ 1º`, `§ 2º` juntos
  - `parent_id`: `f"{doc_id}_art{art_num:03d}"`
  - Campos do payload: todos os metadados + `chunk_type="parent"`, `art_num=N`
  - **NÃO será indexado vetorialmente** — apenas armazenado no payload dos children

- [x] **4.4 — Chunks Child (subdivisão do artigo)**
  - Função `build_child_chunks(doc_row, parent: dict) -> list[dict]`
  - Estratégia de divisão por parágrafo:
    1. Detectar `§ Xº` → cada parágrafo vira um child
    2. Detectar incisos `[IVX]+\s*—` → agrupar em pares (2 incisos por child)
    3. Se não há parágrafos/incisos: RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40, separators=["\n\n", "\n", ". ", " "])
  - Tamanho alvo: 100–200 tokens por child (contar com tiktoken)
  - Prefixo em cada child: `{TIPO} {NUMERO}/{ANO} — {ementa[:150]}\n{TIPO} {NUMERO}/{ANO}, Art. {N}\n`
  - Payload: todos os metadados + `chunk_type="child"`, `parent_id`, `texto_parent` (texto completo do parent)
  - Máximo de 20 children por parent — se exceder, agrupar incisos em grupos de 3

- [x] **4.5 — Fallback para documentos sem artigos**
  - Função `build_fallback_chunks(doc_row, full_text: str) -> list[dict]`
  - Aplicar `RecursiveCharacterTextSplitter`:
    - `chunk_size=500` tokens (parent fallback)
    - `chunk_overlap=80` tokens
    - separators: `["\n\n", "\n", ". ", ", ", " "]`
  - Cada bloco vira tanto parent quanto child (mesmo texto, `parent_id = chunk_id`)
  - `chunk_type="fallback"` — usar para filtragem em análises de qualidade
  - Despachos (DSP) quase sempre usam este caminho

- [x] **4.6 — Prefixo contextual universal**
  - Função `add_contextual_prefix(chunk_text: str, doc_row: pd.Series, art_num: Optional[int] = None) -> str`
  - Formato do prefixo:
    ```
    {TIPO} {NUMERO}/{ANO} — {ementa[:150]}
    [Art. {N} —] (se aplicável)
    ---
    {chunk_text}
    ```
  - Garantir que o prefixo não seja contado nos limites de token do chunk (prefixo adicional, não substitui)

- [x] **4.7 — Geração de chunk_id único**
  - Função `generate_chunk_id(doc_id: str, chunk_type: str, seq: int) -> str`
  - Formato: `f"{doc_id}_{chunk_type[:1]}{seq:04d}"` (ex: `ren20211000_c0001`, `ren20211000_e0000`)
  - UUID determinístico: `hashlib.md5(f"{doc_id}_{chunk_type}_{seq}".encode()).hexdigest()[:16]`

- [x] **4.8 — Pipeline completo de chunking**
  - Função `chunk_document(metadata_row, parsed_text: str) -> list[dict]`
  - Ordem de operações:
    1. Gerar chunk de ementa (sempre)
    2. Detectar estrutura (artigos?)
    3. Se artigos: gerar parents + children por artigo
    4. Senão: gerar chunks fallback
    5. Adicionar prefixo contextual em todos
    6. Calcular token count final (tiktoken)
  - Output: lista de dicts com payload completo pronto para Qdrant

- [x] **4.9 — Processamento batch e salvamento**
  - Processar todos os documentos do `parsed_docs.parquet`
  - Barra de progresso com `tqdm`
  - Salvar `data/processed/chunks.parquet` com todos os chunks
  - Colunas: `chunk_id`, `doc_id`, `tipo_sigla`, `ano`, `chunk_type`, `parent_id`, `art_num`, `texto`, `texto_parent`, `token_count`, todos os metadados do documento
  - Gerar estatísticas: `data/processed/chunk_stats.json`
    ```json
    {
      "total_chunks": 85000,
      "por_tipo": {"ementa": 18688, "child": 55000, "fallback": 8000, "parent_stored_only": 3312},
      "token_stats": {"mean": 145, "median": 130, "p95": 220, "max": 800}
    }
    ```

- [x] **4.10 — Testes do módulo**
  - `tests/test_chunk.py` (implementar TODOs existentes):
    - `test_ementa_chunk_always_created`: todo documento gera exatamente 1 ementa
    - `test_artigo_detection`: texto com "Art. 1º ... Art. 2º" detecta 2 artigos
    - `test_parent_child_link`: children têm `parent_id` correto, `texto_parent` não vazio
    - `test_fallback_for_despacho`: texto curto sem artigos usa fallback, `chunk_type="fallback"`
    - `test_contextual_prefix`: todo chunk começa com `{TIPO} {NUM}/{ANO}`
    - `test_token_limits_parent`: parent não ultrapassa 800 tokens
    - `test_token_limits_child`: child não ultrapassa 220 tokens
    - `test_chunk_id_uniqueness`: todos os chunk_ids são únicos no dataset completo

---

## Módulo 5 — Embedding e Indexação no Qdrant (`src/05_embed_index.py`)

**Objetivo:** Gerar embeddings densos e esparsos para todos os chunks `child` e `ementa`, e indexar no Qdrant local com payload completo.

- [x] **5.1 — Configuração do Qdrant local**
  - Função `get_qdrant_client(path: str) -> QdrantClient`
  - Modo embarcado: `QdrantClient(path="qdrant_db/")`
  - Criar collection `aneel_chunks` se não existir:
    ```python
    client.create_collection(
        collection_name="aneel_chunks",
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
    )
    ```
  - Índices de payload para filtros rápidos:
    - `tipo_sigla` (keyword)
    - `ano` (integer)
    - `chunk_type` (keyword)
    - `is_ementa` (boolean)
    - `revogada` (boolean)

- [x] **5.2 — Carregamento do modelo bge-m3**
  - Função `load_embedding_model() -> BGEM3FlagModel`
  - Modelo: `BAAI/bge-m3` via `FlagEmbedding`
  - Configuração:
    ```python
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,          # metade da VRAM
        device="mps",           # Mac M4; auto-detectar: "cuda" | "mps" | "cpu"
    )
    ```
  - Detectar dispositivo automaticamente: `torch.backends.mps.is_available()` → MPS, `torch.cuda.is_available()` → CUDA, else CPU
  - Logar: modelo carregado, device, VRAM/RAM disponível

- [x] **5.3 — Geração de embeddings em batch**
  - Função `embed_batch(texts: list[str], model: BGEM3FlagModel, batch_size: int = 32) -> dict`
  - Usar `model.encode(texts, batch_size=batch_size, return_dense=True, return_sparse=True, return_colbert_vecs=False)`
  - Output: `{"dense": ndarray[N, 1024], "sparse": list[dict{indices, values}]}`
  - Batch size padrão: 32 (ajustar para 16 se OOM em CPU)
  - Progress bar com `tqdm`

- [x] **5.4 — Montagem dos pontos do Qdrant**
  - Função `build_qdrant_points(chunks: list[dict], embeddings: dict) -> list[PointStruct]`
  - Para cada chunk: criar `PointStruct` com:
    - `id`: int hash do `chunk_id` (MD5 → int)
    - `vector`: `{"dense": dense_vec.tolist(), "sparse": SparseVector(indices=..., values=...)}`
    - `payload`: todos os campos do chunk exceto `chunk_id` (armazenado como `chunk_id` no payload também)
  - **Incluir `texto_parent` no payload** mesmo sendo texto longo — isso é crítico para o parent lookup

- [x] **5.5 — Upload para Qdrant com upsert**
  - Função `upload_to_qdrant(client, points: list[PointStruct], batch_size: int = 100) -> None`
  - Usar `client.upsert(collection_name="aneel_chunks", points=batch)` em lotes de 100
  - Retry automático: 3 tentativas com backoff exponencial (1s, 2s, 4s)
  - Progress bar por lote
  - Ao final: `client.update_collection(...)` para forçar otimização de segmentos

- [x] **5.6 — Filtro de chunks a indexar**
  - Indexar apenas chunks com `chunk_type in ["child", "ementa", "fallback"]`
  - **NÃO indexar** chunks com `chunk_type="parent"` — eles existem apenas no payload
  - Verificar se chunk já existe no Qdrant (pelo `chunk_id`) antes de re-embeder
  - Suporte a modo incremental: `--incremental` flag que pula doc_ids já indexados

- [x] **5.7 — Estatísticas de indexação**
  - Após indexação: `client.get_collection("aneel_chunks").vectors_count`
  - Salvar `data/processed/index_stats.json` atualizado:
    ```json
    {
      "indexed_chunks": 73000,
      "by_type": {"ementa": 18688, "child": 47000, "fallback": 7312},
      "collection_size_mb": 450,
      "indexing_time_minutes": 35
    }
    ```

- [x] **5.8 — Testes do módulo**
  - `tests/test_embed_index.py`:
    - `test_qdrant_collection_created`: collection existe após indexação
    - `test_point_count_matches`: número de pontos == número de chunks child+ementa+fallback
    - `test_payload_has_texto_parent`: child points têm `texto_parent` não vazio
    - `test_dense_vector_dimension`: dimensão = 1024
    - `test_sparse_vector_non_empty`: sparse vector tem ao menos 10 índices
    - `test_ementa_boost_field`: pontos de ementa têm `is_ementa=True` no payload

---

## Módulo 6 — Pipeline de Consulta (`src/06_query.py`)

**Objetivo:** Implementar o pipeline completo de resposta a perguntas: query processing → embedding → hybrid search → RRF → parent lookup → reranking → geração com Claude Sonnet.

- [ ] **6.1 — Query Processing**
  - Função `process_query(question: str) -> dict`
  - Extrair filtros da pergunta via regex simples:
    - Ano: `r'\b(2016|2021|2022)\b'` → `filter_ano`
    - Tipo: `r'\b(REN|REH|REA|DSP|PRT)\b'` → `filter_tipo`
    - Número: `r'\b(?:REN|REH|REA|DSP|PRT)\s*(\d+)\b'` → `filter_numero`
  - Reescrita da query: remover stop words específicas de legislação ("a resolução", "qual é o", "me diga")
  - Output: `{"original": str, "clean": str, "filters": {"ano": int, "tipo": str, "numero": int}}`

- [ ] **6.2 — Embedding da query**
  - Função `embed_query(query: str, model: BGEM3FlagModel) -> dict`
  - Usar `model.encode([query], return_dense=True, return_sparse=True)`
  - Output: `{"dense": list[float], "sparse": SparseVector}`
  - Cache LRU simples para queries idênticas (evitar re-embedding em avaliação)

- [ ] **6.3 — Hybrid Search no Qdrant**
  - Função `hybrid_search(client, dense_vec, sparse_vec, filters: dict, top_k: int = 40) -> list`
  - Busca densa:
    ```python
    client.search(
        collection_name="aneel_chunks",
        query_vector=("dense", dense_vec),
        query_filter=build_filter(filters),
        limit=top_k,
        with_payload=True,
        score_threshold=0.3,
    )
    ```
  - Busca esparsa:
    ```python
    client.search(
        collection_name="aneel_chunks",
        query_vector=("sparse", sparse_vec),
        query_filter=build_filter(filters),
        limit=top_k,
        with_payload=True,
    )
    ```
  - Construção de filtros: função `build_filter(filters: dict) -> Filter` do qdrant_client
  - Executar as duas buscas em paralelo com `asyncio` ou `ThreadPoolExecutor`

- [ ] **6.4 — Fusão RRF (Reciprocal Rank Fusion)**
  - Função `reciprocal_rank_fusion(dense_results, sparse_results, k: int = 60) -> list`
  - Fórmula: `score_rrf(d) = Σ 1 / (k + rank(d, list_i))`
  - Parâmetro `k=60` (padrão da literatura)
  - Boost para ementas: multiplicar score RRF por 1.3 se `is_ementa=True`
  - Retornar top-20 resultados ordenados por score RRF

- [ ] **6.5 — Parent Lookup**
  - Função `lookup_parents(results: list) -> list[dict]`
  - Para cada resultado:
    - Se `chunk_type == "ementa"`: usar o próprio texto da ementa como contexto
    - Se `chunk_type == "child"` ou `"fallback"`: recuperar `texto_parent` do payload
  - Deduplicar por `parent_id` — se dois children do mesmo parent aparecerem, manter apenas uma entrada com o parent
  - Output: lista de dicts com `texto_parent`, `doc_id`, todos os metadados, `rrf_score`
  - Máximo: top-20 contexts únicos

- [ ] **6.6 — Reranking com bge-reranker-v2-m3**
  - Função `rerank(query: str, contexts: list[dict], top_n: int = 5) -> list[dict]`
  - Carregar modelo: `FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="mps")`
  - Criar pares: `[(query, ctx["texto_parent"]) for ctx in contexts]`
  - Chamar: `scores = reranker.compute_score(pairs, normalize=True)`
  - Ordenar contexts por score decrescente, retornar top-5
  - Logar scores: útil para debug de qualidade

- [ ] **6.7 — Formatação do contexto para o LLM**
  - Função `format_context(contexts: list[dict]) -> str`
  - Formato de cada contexto:
    ```
    [FONTE 1] {TIPO} {NUMERO}/{ANO}
    Data: {data_pub} | Situação: {situacao}
    {texto_parent}
    ---
    ```
  - Total máximo: 8000 tokens (usando tiktoken para medir)
  - Se exceder: truncar os contexts de menor score

- [ ] **6.8 — Geração com Claude Sonnet**
  - Função `generate_answer(question: str, contexts: str, client: anthropic.Anthropic) -> str`
  - Modelo: `claude-sonnet-4-5` (ou o mais recente disponível)
  - System prompt:
    ```
    Você é um especialista em legislação do setor elétrico brasileiro.
    Responda apenas com base nas fontes fornecidas.
    Cite SEMPRE as fontes no formato [TIPO NUMERO/ANO, Art. X] inline na resposta.
    Se a informação não estiver nas fontes, diga explicitamente que não encontrou.
    Responda em português formal.
    ```
  - Temperatura: 0.1 (respostas determinísticas)
  - max_tokens: 1500
  - Tratamento de erro: `anthropic.APIError` → retry 2x com backoff

- [ ] **6.9 — Interface CLI de consulta**
  - Script `src/06_query.py` executável standalone
  - Modo interativo: `python src/06_query.py` → loop de perguntas
  - Modo single: `python src/06_query.py --question "O que é microgeração distribuída?"`
  - Flag `--debug`: mostrar chunks recuperados, scores RRF, scores reranker
  - Flag `--no-rerank`: pular reranking (útil para comparar qualidade)

- [ ] **6.10 — Testes do módulo**
  - `tests/test_query.py`:
    - `test_filter_extraction_ano`: query com "2021" extrai filtro correto
    - `test_filter_extraction_tipo`: query com "REN" extrai tipo correto
    - `test_rrf_boost_ementa`: ementa recebe score maior que child equivalente
    - `test_parent_deduplication`: dois children do mesmo parent viram 1 context
    - `test_answer_has_citation`: resposta contém pelo menos 1 citação `[...]`
    - `test_source_not_hallucinated`: todas as citações correspondem a docs no context

---

## Módulo 7 — Avaliação com RAGAS (`src/07_evaluate.py`)

**Objetivo:** Medir a qualidade do pipeline usando o golden set anotado. Gerar relatório completo de métricas.

- [ ] **7.1 — Expansão do golden set**
  - Arquivo `tests/golden_set.json` com mínimo de 20 perguntas anotadas
  - Cobertura obrigatória de tipos:
    - 5x perguntas sobre definições em REN (microgeração, PRODIST, etc.)
    - 5x perguntas sobre metadados diretos (situação, data, autor)
    - 5x perguntas agregadas (quais distribuidoras, quais REH de 2022)
    - 5x perguntas sobre tabelas tarifárias (requerem extração de REH)
  - Cada entrada: `{"question": str, "expected_answer": str, "relevant_doc_ids": [str], "notes": str}`
  - Anotar `relevant_doc_ids` com os doc_ids reais dos documentos relevantes

- [ ] **7.2 — Execução do pipeline sobre o golden set**
  - Para cada pergunta do golden set: executar pipeline completo (Módulo 6)
  - Coletar: `answer`, `contexts` (textos), `context_doc_ids`
  - Salvar em `results/eval_run_{timestamp}.json`:
    ```json
    {
      "question": str,
      "expected_answer": str,
      "generated_answer": str,
      "retrieved_contexts": [str],
      "retrieved_doc_ids": [str],
      "relevant_doc_ids": [str]
    }
    ```

- [ ] **7.3 — Métricas RAGAS**
  - Usar `ragas` >= 0.1 com métricas:
    - `faithfulness`: a resposta se sustenta no contexto? (LLM judge)
    - `answer_relevancy`: a resposta é relevante para a pergunta? (embedding similarity)
    - `context_precision`: os contextos recuperados são relevantes? (LLM judge)
    - `context_recall`: os contextos cobrem a resposta esperada? (LLM judge)
  - LLM judge: Claude Haiku (mais barato) ou `gpt-4o-mini`
  - Configuração RAGAS:
    ```python
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    ```

- [ ] **7.4 — Métricas de retrieval diretas**
  - `Recall@K` para K=5, 10, 20: fração das perguntas onde o doc relevante aparece no top-K
  - `MRR` (Mean Reciprocal Rank): posição média do primeiro doc relevante
  - `Hit Rate`: fração de perguntas com ao menos 1 doc relevante no top-5
  - Calcular separadamente para: ementas, children, fallback

- [ ] **7.5 — Ablação de componentes**
  - Comparar: dense-only vs sparse-only vs hybrid (RRF)
  - Comparar: sem reranking vs com reranking
  - Comparar: sem boost de ementa vs com boost de ementa
  - Salvar resultados de ablação em `results/ablation_{timestamp}.json`

- [ ] **7.6 — Relatório final de avaliação**
  - Gerar `results/eval_report_{timestamp}.md` com:
    - Tabela de métricas RAGAS
    - Tabela de métricas de retrieval
    - Tabela de ablação
    - Top-3 perguntas com pior performance (análise qualitativa)
    - Recomendações de melhoria

- [ ] **7.7 — Testes do módulo**
  - `tests/test_evaluate.py`:
    - `test_golden_set_valid`: golden set tem pelo menos 10 perguntas com `expected_answer` não vazio
    - `test_eval_output_schema`: output de avaliação tem todos os campos necessários
    - `test_metrics_in_range`: todas as métricas entre 0 e 1
    - `test_retrieval_recall_baseline`: Recall@10 > 0.5 (sanidade mínima)

---

## Módulo 8 — Utilitários e Qualidade (`src/utils/`)

**Objetivo:** Funções auxiliares compartilhadas por múltiplos módulos. Não são executáveis standalone.

- [ ] **8.1 — Contagem de tokens (`src/utils/token_counter.py`)**
  - Função `count_tokens(text: str, model: str = "cl100k_base") -> int`
  - Usar `tiktoken.get_encoding(model).encode(text)`
  - Função `truncate_to_tokens(text: str, max_tokens: int) -> str`

- [ ] **8.2 — Normalização de texto (`src/utils/text_utils.py`)**
  - `normalize_numero(text: str) -> str`: "REN 1.000/2021" → "REN 1000/2021"
  - `normalize_tipo(raw: str) -> str`: "Resolução Normativa" → "REN"
  - `clean_whitespace(text: str) -> str`: múltiplos espaços/newlines → único
  - `remove_control_chars(text: str) -> str`: remove `\x00-\x1f` exceto `\n\t`
  - `normalize_encoding(text: str) -> str`: corrige caracteres mal-encodados (latin-1 → utf-8)

- [ ] **8.3 — Filtros Qdrant (`src/utils/qdrant_filters.py`)**
  - Função `build_filter(filters: dict) -> Optional[Filter]`
  - Suporte: `ano` (int), `tipo_sigla` (str), `revogada` (bool), `chunk_type` (str)
  - Combinação AND de múltiplos filtros

- [ ] **8.4 — Hash e IDs (`src/utils/ids.py`)**
  - `doc_id_from_filename(filename: str) -> str`
  - `chunk_id_hash(doc_id: str, chunk_type: str, seq: int) -> str`
  - `int_id_from_str(s: str) -> int` (para IDs do Qdrant)

- [ ] **8.5 — Verificação de ambiente (`src/utils/env_check.py`)**
  - Verificar: Python >= 3.11, dependências instaladas, `ANTHROPIC_API_KEY` setado, Qdrant abrível, modelo bge-m3 baixado
  - Detectar device: MPS (M1/M2/M4) > CUDA > CPU
  - Estimar VRAM/RAM disponível

---

## Módulo 9 — Notebooks de Exploração (`notebooks/`)

**Objetivo:** Análise exploratória e debugging interativo.

- [ ] **9.1 — `notebooks/01_explore_metadata.ipynb`**
  - Carregar `metadata.parquet`, explorar distribuição de tipos, anos, tamanho de ementas
  - Plotar histogramas: quantidade por tipo/ano, comprimento de ementa
  - Verificar completude: % com PDF baixado, % com situação definida

- [ ] **9.2 — `notebooks/02_inspect_chunks.ipynb`**
  - Carregar `chunks.parquet`, inspecionar exemplos de cada tipo de chunk
  - Verificar prefixos contextuais, distribuição de tokens
  - Comparar chunk de ementa vs chunk de artigo do mesmo documento
  - Visualizar exemplo completo: ementa → parent → 3 children

- [ ] **9.3 — `notebooks/03_query_debug.ipynb`**
  - Interface interativa para testar queries passo a passo
  - Mostrar: query processada, filtros extraídos, top-10 results antes e depois do reranking
  - Comparar scores: dense, sparse, RRF, reranker
  - Útil para diagnóstico de falhas de retrieval

- [ ] **9.4 — `notebooks/04_eval_analysis.ipynb`**
  - Carregar `results/eval_run_*.json`, analisar erros
  - Identificar padrões nas perguntas com baixo score
  - Visualizar curva Precision@K e Recall@K
  - Comparar ablações side-by-side

---

## Módulo 10 — Documentação e Entrega (`docs/`)

**Objetivo:** Documentação técnica completa do sistema para apresentação ao grupo de estudos.

- [ ] **10.1 — `docs/architecture.md`**
  - Descrição detalhada de cada componente do pipeline
  - Diagrama ASCII do fluxo offline e online
  - Justificativa de cada escolha técnica (por que bge-m3, por que RRF, etc.)

- [ ] **10.2 — `docs/chunking_strategy.md`**
  - Explicação detalhada da estratégia Parent-Child Document-Aware
  - Exemplos reais de chunks gerados de documentos ANEEL
  - Comparação com chunking simples (por tamanho fixo)

- [ ] **10.3 — `README.md`**
  - Instruções de instalação e configuração
  - Como rodar cada módulo
  - Como executar avaliação
  - Resultados obtidos (tabela de métricas)

- [ ] **10.4 — `docs/results.md`**
  - Métricas finais do sistema
  - Comparação com baselines simples
  - Análise de pontos fortes e fracos
  - Próximos passos

---

## Status Geral

| Módulo | Descrição | Status |
|--------|-----------|--------|
| 0 | Infraestrutura e Ambiente | ✅ Concluído |
| 1 | Consolidação de Metadados | ✅ Concluído |
| 2 | Download de PDFs | ✅ Concluído (externo) |
| 3 | Parsing de PDFs | ✅ Concluído |
| 4 | Chunking Document-Aware | ✅ Concluído |
| 5 | Embedding e Indexação | ✅ Concluído |
| 6 | Pipeline de Consulta | ⬜ Pendente |
| 7 | Avaliação com RAGAS | ⬜ Pendente |
| 8 | Utilitários | ⬜ Pendente |
| 9 | Notebooks | ⬜ Pendente |
| 10 | Documentação | ⬜ Pendente |

---

## Ordem de Implementação Recomendada

```
0 → 1 → 2 → 3 → 4 → 8 (junto com 4) → 5 → 6 → 7 → 9 → 10
```

Prioridade máxima: **Módulos 1, 3, 4, 5, 6** — são o núcleo do pipeline.  
Módulo 7 deve ser executado após cada melhoria incremental para medir impacto.

---

*Última atualização: 2026-04-22*
