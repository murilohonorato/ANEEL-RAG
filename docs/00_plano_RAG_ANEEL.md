# Desafio RAG — Legislação ANEEL
## Estudo da arquitetura e plano de processamento/chunking

> **Objetivo do desafio:** construir um sistema agêntico com foco em RAG capaz de responder perguntas sobre documentos reais do setor elétrico brasileiro (legislação ANEEL 2016, 2021, 2022). A avaliação é um benchmark de perguntas anotado por especialista. **O foco é a qualidade do pipeline de recuperação**, não a parte agêntica.

---

## 1. O que é RAG, afinal

RAG (Retrieval-Augmented Generation) é um padrão onde o LLM não responde apenas a partir de seus pesos, mas recebe **contexto externo recuperado em tempo de inferência**. Isso resolve três problemas estruturais do LLM puro:

1. **Conhecimento desatualizado ou fora do treino** — resoluções da ANEEL de 2022 não estão confiáveis no modelo.
2. **Alucinações em domínios especializados** — linguagem jurídica/regulatória exige ancoragem no texto-fonte.
3. **Rastreabilidade** — no setor elétrico é preciso citar artigo, inciso, portaria; o RAG permite retornar a fonte.

O pipeline RAG clássico tem duas fases temporalmente separadas:

```
OFFLINE (ingestão/indexação)          ONLINE (consulta)
─────────────────────────────         ─────────────────
Fonte → Parsing → Normalização        Pergunta do usuário
     → Chunking → Embeddings                 ↓
     → Índice (vetor + lexical)        Reescrita/expansão de query
                                              ↓
                                       Retrieval (top-k)
                                              ↓
                                       Reranking
                                              ↓
                                       Montagem de contexto
                                              ↓
                                       LLM → Resposta + citações
```

Cada seta esconde uma decisão de engenharia. Abaixo eu destrincho cada uma, e depois aterrizo no nosso dataset.

---

## 2. Componentes do pipeline — o que estudar em cada um

### 2.1. Ingestão e parsing
**O que é:** transformar o documento bruto (PDF, HTML, DOCX) em texto estruturado e limpo.

**Por que importa:** chunking e embedding operam sobre *texto*. Se o parser entrega lixo (cabeçalhos repetidos, tabelas quebradas, OCR ruim), todo o resto do pipeline degrada. Em documentos jurídicos brasileiros isso é especialmente crítico porque:
- PDFs antigos frequentemente são **digitalizações escaneadas** → precisam OCR.
- PDFs regulatórios têm **tabelas complexas** (tarifas, MW, coeficientes).
- Têm **numeração hierárquica** (Art. 1º, §2º, inciso III, alínea a) que é *semântica* — preservar essa estrutura melhora muito o retrieval.

**Estratégias/ferramentas:**
| Ferramenta | Quando usar | Observações |
|---|---|---|
| `pypdf` / `pdfplumber` | PDF com texto nativo, layout simples | Rápido, gratuito. `pdfplumber` extrai tabelas melhor. |
| `PyMuPDF` (fitz) | PDF com layout complexo, extração posicional | Dá coordenadas — útil para detectar cabeçalhos/rodapés repetidos. |
| `unstructured.io` | Pipeline turnkey, mistura PDF/HTML/DOCX | Detecta título, parágrafo, tabela automaticamente. |
| `marker`, `docling`, `LlamaParse` | PDFs complexos, layout denso | Usam ML para parsing estruturado; mais lento/custoso. |
| `tesseract` (OCR) | Digitalizações escaneadas | Combinar com `ocrmypdf` para reprocessar só o que precisa. |

**O que avaliar para o nosso dataset:**
- Quantos dos ~27 mil PDFs são escaneados vs. texto nativo? (Amostragem.)
- Existe cabeçalho/rodapé repetitivo do ANEEL a remover?
- Tabelas são importantes para as perguntas do benchmark? (Provavelmente sim para REH — Resoluções Homologatórias de tarifas.)

---

### 2.2. Normalização e limpeza
**O que é:** após extrair o texto, aplicar regras para torná-lo uniforme.

**Decisões típicas:**
- Remover cabeçalhos/rodapés/numeração de página repetida.
- Padronizar hifenização de quebra de linha (`pala-\nvra` → `palavra`).
- Normalizar acentuação e encoding (o PDF ANEEL costuma ter `ligaduras` estranhas).
- Decidir se preserva *case* ou não (embeddings modernos preservam; BM25 geralmente lowercase).
- Extrair e **preservar metadados estruturais**: número da norma, data, órgão, ementa.

**Cuidado:** nunca jogar fora *o que o retrieval vai precisar*. Ex.: o número "REN 1000/2021" é a assinatura do documento — tem que sobreviver.

---

### 2.3. Chunking
**O que é:** dividir o texto em pedaços do tamanho certo para (a) caber no embedding model, (b) ter granularidade suficiente para recuperação, (c) manter coerência semântica.

**Por que é o **coração do problema**:** chunk pequeno demais → perde contexto e o LLM não entende. Chunk grande demais → embedding fica "genérico" e o retrieval traz junk. Chunk no lugar errado → corta um artigo no meio e o trecho retornado não faz sentido sem o de cima.

**Estratégias, do mais ingênuo ao mais sofisticado:**

1. **Fixed-size chunking** — N tokens com overlap de M tokens. Simples, baseline útil. Ignora estrutura.
2. **Recursive character splitting** (LangChain `RecursiveCharacterTextSplitter`) — tenta quebrar por `\n\n`, depois `\n`, depois `.`, depois caractere. Melhor que fixed, ainda cego à estrutura.
3. **Document-aware chunking** — quebra por **unidade lógica do documento**. Para normas jurídicas, isso é ouro: chunk = um artigo, ou artigo + parágrafos. É o que eu recomendo aqui.
4. **Semantic chunking** — usa embeddings para detectar mudança de tópico dentro do texto e quebrar ali. Bom para texto narrativo; em norma jurídica rende pouco porque a estrutura já é explícita.
5. **Late chunking / contextual retrieval** — embedar o documento inteiro primeiro (modelos de long-context) e depois segmentar, ou usar a técnica da Anthropic de *contextual retrieval* (prefixar cada chunk com um resumo do doc). Ganho documentado grande em retrieval quality.
6. **Hierarchical chunking (parent-child)** — indexa chunks pequenos mas retorna o chunk "pai" (seção inteira). Melhor dos dois mundos.

**Tamanhos típicos (tokens):** 200–500 tokens por chunk, overlap 10–20%. Para normas jurídicas, prefiro chunk variável (o tamanho do artigo) com um cap de 800 tokens.

**Metadados por chunk** (fundamental): `{tipo_doc, numero, ano, data_publicacao, orgao, titulo, artigo, url_pdf}`. São usados para **filtrar** no retrieval ("só 2022", "só REN") e para **citação** na resposta.

---

### 2.4. Embeddings
**O que é:** transformar cada chunk em um vetor numérico que captura seu significado.

**Decisões:**
- **Modelo:** tem que funcionar em **português jurídico**. Candidatos:
  - OpenAI `text-embedding-3-large` (3072d) — excelente, multilíngue, custo ok.
  - `intfloat/multilingual-e5-large` (1024d) — open, muito bom em PT, roda local.
  - `BAAI/bge-m3` (1024d) — open, multilíngue, suporta hybrid (dense + sparse + multi-vector).
  - Cohere `embed-multilingual-v3` — pago, ótimo reranker companheiro.
- **Dimensionalidade:** mais = mais memória/compute, nem sempre mais qualidade. `bge-m3` com 1024d é um sweet spot.
- **Normalização L2:** quase sempre sim, para usar cosine similarity.
- **Re-embedding ao mudar modelo:** o índice é "casado" com o modelo; não dá para misturar.

---

### 2.5. Indexação
**O que é:** armazenar os vetores (e metadados) numa estrutura que permita busca aproximada rápida (ANN).

**Opções:**
| Vector DB | Perfil | Observação |
|---|---|---|
| **ChromaDB** | Local, embarcado, simples | Ótimo para protótipo, roda em processo. |
| **Qdrant** | Local ou cloud, muito performático | Filtros por metadados excelentes, open-source. |
| **Weaviate** | Cloud/self-host, hybrid nativo | Tem BM25 integrado. |
| **pgvector** | Postgres + extensão | Ótimo se já usa Postgres, indexação HNSW. |
| **FAISS** | Biblioteca pura, sem persistência bonita | Baseline acadêmico. |
| **LanceDB** | Colunar, serverless | Bom para datasets grandes. |

**Hybrid search (muito importante aqui):** além do vetor denso, manter um índice **lexical (BM25)**. Pergunta como *"o que diz a REN 414/2010?"* — o nome da norma é **token literal**; BM25 encontra perfeito, embedding pode falhar. A combinação BM25 + denso (fusão via **RRF — Reciprocal Rank Fusion**) é hoje o default de quem quer qualidade.

---

### 2.6. Query processing
**O que é:** não mandar a pergunta crua direto pro retrieval.

**Técnicas:**
- **Query rewriting** — LLM reformula a pergunta para ficar melhor "embedável" (remove conversational fluff).
- **Query expansion / HyDE** — LLM gera uma "resposta hipotética" e embeda *ela*, não a pergunta. Captura vocabulário da fonte.
- **Multi-query** — LLM gera 3–5 variantes da pergunta; une os resultados.
- **Decomposição** — pergunta complexa quebrada em sub-perguntas (retrieval para cada).
- **Extração de filtros** — LLM identifica "ano 2022" na pergunta e passa como filtro de metadado, não como conteúdo semântico.

Para o nosso caso, **extração de filtros** é particularmente valiosa: perguntas provavelmente mencionam ano, tipo de norma (resolução, despacho) ou número.

---

### 2.7. Retrieval
- Top-k (tipicamente k=20–50 na primeira passada).
- Filtros por metadados.
- Fusão de scores se hybrid.

### 2.8. Reranking
**O que é:** reordenar os top-k usando um modelo mais caro e mais preciso (**cross-encoder**) e ficar só com os top-n (n=3–8).

**Por que:** embedding é um bi-encoder, "comprime" pergunta e doc separadamente. Cross-encoder vê os dois juntos e mede relevância direta — muito mais preciso, mas lento (roda só nos candidatos).

**Modelos:** `Cohere Rerank 3.5`, `BAAI/bge-reranker-v2-m3`, `jina-reranker-v2`.

**Efeito:** frequentemente é o **single biggest win** depois de um embedding decente.

### 2.9. Montagem de contexto e geração
- Compor o prompt com os chunks selecionados (ordenados, com separadores claros).
- Incluir instruções anti-alucinação: *"responda apenas com base nos trechos; se não houver informação, diga que não sabe"*.
- **Forçar citação**: cada afirmação precisa indicar o chunk/documento-fonte.
- Escolher o LLM (Claude, GPT-4o, Llama 70B) — para PT jurídico, Claude costuma ter vantagem.

### 2.10. Avaliação
Não é opcional — é como o desafio vai ser julgado.

- **Métricas de retrieval**: Recall@k, MRR, nDCG — precisam de um *ground-truth* (quais chunks são relevantes).
- **Métricas end-to-end**: faithfulness (a resposta se sustenta no contexto?), answer relevance, context precision. Frameworks: **RAGAS**, **TruLens**, **DeepEval**.
- **Golden set manual**: 30–100 perguntas nossas, anotadas, para iterar rápido antes do benchmark oficial.

---

## 3. Análise do dataset

Inspecionei os 3 JSONs. Resumo:

| Arquivo | Datas cobertas | Registros | Refs a PDF | Marcados "baixado" |
|---|---|---|---|---|
| 2016 | 366 dias | 4.269 | 6.279 | 6.252 |
| 2021 | 365 dias | 6.993 | 9.624 | 9.621 |
| 2022 | 365 dias | 7.426 | 11.136 | 11.117 |
| **Total** | — | **18.688** | **27.039** | **26.990** |

**Estrutura:** dicionário `{data: {status, registros: [...]}}`. Cada registro traz: `titulo`, `autor`, `material`, `esfera`, `situacao`, `assinatura` (data), `publicacao` (data), `assunto`, `ementa` (resumo oficial!), `pdfs: [{tipo, url, arquivo, baixado}]`.

**Distribuição por tipo de documento (top):**

| Sigla | Nome | % aproximada |
|---|---|---|
| DSP | Despacho | ~53% |
| PRT | Portaria | ~17% |
| REA | Resolução Autorizativa | ~21% |
| REH | Resolução Homologatória | ~2,5% |
| REN | Resolução Normativa | ~0,8% |
| ECT, AAP, EDT, AVS, COM, ACP, ATS, DEC, etc. | outros | <5% |

**Observações críticas:**

1. **Os PDFs NÃO estão na pasta** — o JSON tem URLs (`http://www2.aneel.gov.br/cedoc/...`) e um flag `"baixado": true` que indica que *quem montou o dataset* baixou, não que *nós* temos localmente. **Primeira ação prática: baixar.**
2. **A "ementa" é ouro.** É um resumo oficial curto do que a norma trata. Para muitas perguntas, a ementa + título + assunto já responde — não precisa nem ler o PDF. Estratégia: indexar ementa como chunk *separado e privilegiado*.
3. **Volume real:** ~27 mil PDFs. Se cada um tiver 3–10 páginas em média, são centenas de milhares a ~1 milhão de chunks. Precisa planejar custo de embedding e tempo.
4. **REN (Resolução Normativa)** são as normas "de maior peso" apesar de serem poucas (~150 no total). Perguntas de especialista muito provavelmente focam nelas, nas REH (tarifas) e nas RES (resoluções gerais). Despachos são decisões administrativas pontuais — grande volume, baixa densidade de "conhecimento geral".
5. **Campos estruturados** (data, tipo, número, autor, ementa, URL) devem virar **metadados de cada chunk**, não texto indexado.

---

## 4. Plano concreto — etapas propostas

### Etapa 0 — Setup e exploração (0,5 dia)
- Criar repositório e ambiente (`uv` ou `poetry`).
- Stack recomendada (sem restrição, open-first, com fallback pago):
  - **Parsing:** `pdfplumber` + `PyMuPDF` (fallback `unstructured` para casos difíceis).
  - **Embedding:** `BAAI/bge-m3` local (via `sentence-transformers` ou `FastEmbed`). Fallback: `text-embedding-3-large`.
  - **Vector DB:** **Qdrant** local (docker). Suporta filtros por payload e integração com sparse.
  - **Lexical:** `rank-bm25` ou o sparse vectors do próprio Qdrant.
  - **Reranker:** `BAAI/bge-reranker-v2-m3` local.
  - **LLM (geração):** Claude Sonnet via API (pt-br jurídico bom) — ou GPT-4o.
  - **Avaliação:** `RAGAS`.
  - **Orquestração:** código próprio simples primeiro; `LangGraph` se o requisito agêntico apertar.

### Etapa 1 — Consolidação dos metadados (0,5 dia)
- Carregar os 3 JSONs e unificar num DataFrame/Parquet com colunas normalizadas: `doc_id, ano, data_publicacao, data_assinatura, tipo_sigla, tipo_nome, numero, titulo, autor, esfera, situacao, assunto, ementa, pdf_url, pdf_arquivo, pdf_tipo`.
- Gerar `doc_id` estável (ex.: hash de `arquivo`).
- Validar: duplicatas, URLs quebradas, campos faltantes, parsing das datas.
- **Entregável:** `metadata.parquet` com ~27 mil linhas (uma por PDF).

### Etapa 2 — Download e cache de PDFs (0,5–1 dia, batch)
- Download concorrente (async, ~20 workers) com retry exponencial.
- Armazenar em `pdfs/{ano}/{arquivo}.pdf`.
- Registrar tamanho, hash, mime, status em `metadata.parquet`.
- Separar os que falharam para tratamento manual.
- **Risco:** servidor da ANEEL pode rate-limitar; rodar com sleep/jitter.

### Etapa 3 — Parsing dos PDFs (1–2 dias)
- **Pipeline de decisão por PDF:**
  1. Tentar `PyMuPDF` → se tamanho do texto extraído/página < limiar (ex.: 100 chars), marcar como escaneado.
  2. Para escaneados, rodar `ocrmypdf` (Tesseract, idioma `por`) e reextrair.
  3. Remover cabeçalho/rodapé repetitivo detectando linhas que aparecem em >70% das páginas.
  4. Limpar hifenização de fim de linha, normalizar espaços.
- Salvar texto limpo em `text/{doc_id}.txt` + `text/{doc_id}.meta.json` (páginas, OCR sim/não, qualidade estimada).
- **Amostragem de qualidade:** inspecionar 30 PDFs aleatórios de tipos diferentes antes de escalar.

### Etapa 4 — Chunking (a decisão principal — 1 dia)
**Estratégia escolhida: híbrida, *document-aware* com fallback recursivo + contextual prefix.**

**Pseudocódigo:**
```
para cada documento:
    texto = texto_limpo
    # 1. Tentar segmentação estrutural
    blocos = detectar_artigos_e_paragrafos(texto)
        # regex: r'^\s*Art\.?\s*\d+[ºo°]?\.?\s'  e  §, inciso, alínea
    se len(blocos) >= 2:
        chunks = agrupar(blocos, max_tokens=600, min_tokens=80)
        # junta Art. 1º + seus §§ até o cap
    senão:
        # despacho curto, PDF sem estrutura clara
        chunks = recursive_split(texto, size=500, overlap=80)

    # 2. Enriquecer cada chunk com um "prefixo contextual"
    prefixo = f"{tipo_nome} {numero}/{ano} — {ementa[:200]}"
    chunks = [f"{prefixo}\n\n{c}" para c em chunks]

    # 3. Chunk especial "cabeçalho"
    chunks.insert(0, f"{titulo}\nAutor: {autor}\nAssunto: {assunto}\nEmenta: {ementa}")
```

**Justificativa:**
- Preservar a unidade "artigo" mantém coerência jurídica no retrieval.
- O prefixo contextual implementa a ideia do *contextual retrieval* da Anthropic, barata e eficaz aqui (já temos título+ementa prontos no JSON, não precisa LLM resumir).
- O chunk especial cabeçalho/ementa tende a ser o mais recuperado para perguntas gerais ("do que trata a REN 1000?").

**Metadados por chunk (payload do Qdrant):**
```json
{
  "doc_id": "...",
  "tipo_sigla": "REN",
  "tipo_nome": "Resolução Normativa",
  "numero": "1000",
  "ano": 2021,
  "data_publicacao": "2021-12-07",
  "titulo": "...",
  "ementa": "...",
  "assunto": "...",
  "pdf_url": "...",
  "chunk_idx": 5,
  "chunk_type": "artigo" | "ementa" | "bloco_generico",
  "art_num": 12  // se aplicável
}
```

**Entregável:** `chunks.parquet` + um relatório (histograma de tamanho em tokens, chunks por tipo de norma, distribuição por ano).

### Etapa 5 — Embedding e indexação (0,5–1 dia de compute)
- Rodar `bge-m3` em batch (GPU se possível — senão CPU com `FastEmbed` ONNX).
- Popular Qdrant com dense + sparse vectors + payload (metadados).
- Configurar índice HNSW (`m=16, ef_construct=200`).
- **Smoke tests:** 10 perguntas fabricadas cobrindo diferentes tipos/anos; inspecionar os top-10.

### Etapa 6 — Pipeline de query (1 dia)
```
pergunta → LLM(extrai filtros: ano, tipo, número)
        → LLM(query rewrite + HyDE opcional)
        → busca dense (Qdrant) + busca sparse (BM25/Qdrant) com filtros
        → fusão RRF → top-30
        → reranker (bge-reranker-v2-m3) → top-5
        → monta prompt com chunks + instrução de citação
        → LLM gera resposta com [doc_id] em cada afirmação
```

### Etapa 7 — Avaliação e iteração (contínuo)
- Montar um **golden set interno** (30–50 perguntas minhas, com respostas esperadas e chunks relevantes).
- Rodar RAGAS (faithfulness, context precision/recall).
- Iterar sobre: tamanho de chunk, presença/ausência do prefixo contextual, k do retrieval, com/sem reranker, com/sem HyDE.
- Registrar cada experimento num CSV/MLflow.

### Etapa 8 — Agente (só depois do RAG ficar bom)
Dado que o desafio diz explicitamente que o **foco não é o agente**, manter o agente mínimo:
- Uma tool `search(query, filters)` que encapsula o pipeline acima.
- Uma tool `fetch_document(doc_id)` para quando o agente quiser o documento inteiro.
- Loop simples: o LLM decide se precisa buscar mais, refinar a query, ou responder.

---

## 5. Riscos e pontos de atenção

1. **OCR em escala** — se muitos PDFs forem escaneados, o parsing pode virar gargalo. Mitigar rodando OCR em paralelo e cacheando.
2. **Custo de embedding** — ~1M de chunks × 1024 dim = viável local com bge-m3 em GPU (horas). Com OpenAI, orçar: 1M × ~500 tokens × $0.13/M ≈ US$65. Aceitável, mas local é grátis.
3. **Vocabulário regulatório** — termos como "MWmédio", "CUSD", "CUST", "CCEE", "REH" precisam sobreviver ao tokenizer. Verificar no smoke test.
4. **Número-da-norma casando errado** — "REN 1000" e "REN 1.000" e "Resolução Normativa 1000/2021" — normalizar antes de indexar e antes de buscar.
5. **Perguntas fora do escopo temporal** — dataset só tem 2016, 2021, 2022. Perguntas sobre 2017–2020 ou 2023+ não têm resposta; o sistema deve reconhecer e dizer isso.
6. **Ambiguidade em despachos** — mesma numeração em anos diferentes; sempre combinar `(tipo, numero, ano)`.
7. **Revogações** — campo `situacao` indica se a norma foi revogada. Uma pergunta "qual a regra atual para X?" deveria priorizar normas *não* revogadas. Vale um filtro/ranking usando isso.

---

## 6. Sugestão de ordem para estudo (você, esta semana)

1. Ler 2 papers/posts curtos e específicos:
   - *Contextual Retrieval* (Anthropic, 2024) — a técnica do prefixo que vamos usar.
   - *Lost in the Middle* (Liu et al.) — onde posicionar chunks no prompt.
2. Dar uma rodada no **RAGAS tutorial** — só pra internalizar as métricas.
3. Rodar manualmente **uma única norma** (escolher 1 REN) ponta-a-ponta: parse → chunk → embed → query. Vai expor 80% das decisões.
4. Depois escalar.

---

## 7. Próximos entregáveis (ordem sugerida)

- [ ] `scripts/01_consolidate_metadata.py` → `metadata.parquet`
- [ ] `scripts/02_download_pdfs.py` → pasta `pdfs/`
- [ ] `scripts/03_parse_pdfs.py` → pasta `text/`
- [ ] `scripts/04_chunk.py` → `chunks.parquet`
- [ ] `scripts/05_embed_and_index.py` → Qdrant populado
- [ ] `scripts/06_query.py` → pipeline de consulta
- [ ] `notebooks/eval.ipynb` → golden set + RAGAS
- [ ] `README.md` com resultados e decisões

Me diga por qual você quer começar e eu já implemento.
