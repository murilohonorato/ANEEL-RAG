# Estratégia de Chunking — ANEEL RAG

## Visão Geral

O pipeline de chunking do ANEEL RAG usa uma estratégia **Parent-Child Document-Aware** que respeita a hierarquia jurídica dos documentos (artigos → parágrafos → incisos). Isso é fundamental para recuperar contextos completos sem perder a relação entre as partes da norma.

---

## Por Que Não Usar Chunking Simples?

O chunking por tamanho fixo (ex.: 512 tokens com overlap de 50) é a abordagem mais comum, mas falha em documentos jurídicos:

| Problema | Chunking fixo | Parent-Child |
|----------|--------------|--------------|
| Artigo cortado no meio | ✅ Frequente | ❌ Nunca |
| Parágrafo sem contexto do artigo | ✅ Frequente | ❌ Nunca |
| Perda de identidade do documento | ✅ Frequente | ❌ Nunca (prefixo) |
| Recuperação imprecisa | ✅ Alta | ❌ Baixa |
| LLM sem contexto legal | ✅ Comum | ❌ Sempre contextualizado |

---

## Os Três Tipos de Chunk

### 1. Chunk de Ementa

Gerado diretamente do JSON de metadados, sem precisar do PDF. Representa a identidade completa do documento.

**Texto do chunk:**
```
REN 1000/2021 — Estabelece os Procedimentos de Distribuição de Energia Elétrica no
Sistema Elétrico Nacional – PRODIST e dá outras providências.

Tipo: Resolução Normativa
Número: 1000
Ano: 2021
Autor: ANEEL
Assunto: Distribuição de energia elétrica
Situação: NÃO CONSTA REVOGAÇÃO EXPRESSA
Data de publicação: 2021-12-07
```

**Características:**
- 1 por documento, sempre (mesmo sem PDF baixado)
- ~100–200 tokens
- Recebe boost de 1.3× no RRF (ementas são escritas por especialistas)
- `chunk_type = "ementa"`, `is_ementa = True`

**Por que é importante?** Perguntas do tipo "qual a situação da REN 1000?" ou "quem emitiu a REN 482?" são respondidas perfeitamente com a ementa, sem precisar do PDF.

---

### 2. Chunk Parent (não indexado)

Representa um artigo completo com todos os seus parágrafos e incisos.

**Texto do parent:**
```
REN 1000/2021 — Estabelece os Procedimentos de Distribuição...
REN 1000/2021, Art. 7
---
Art. 7º Para os fins desta Resolução, consideram-se:

§ 1º Microgeração distribuída: central geradora de energia elétrica, com potência
instalada menor ou igual a 75 kW e que utilize cogeração qualificada, conforme
regulamentação da ANEEL, ou fontes renováveis de energia elétrica, conectada na rede
de distribuição por meio de instalações de unidades consumidoras.

§ 2º Minigeração distribuída: central geradora de energia elétrica, com potência
instalada superior a 75 kW e menor ou igual a 5 MW para fontes incentivadas ou com
potência instalada superior a 75 kW e menor ou igual a 3 MW para as demais fontes,
e que utilize cogeração qualificada, conforme regulamentação da ANEEL, ou fontes
renováveis de energia elétrica, conectada na rede de distribuição por meio de
instalações de unidades consumidoras.
```

**Características:**
- 400–800 tokens (artigo completo)
- **NÃO é indexado vetorialmente**
- Armazenado no payload do Qdrant (`texto_parent`)
- Recuperado via `parent_id` quando um child correspondente é encontrado

---

### 3. Chunk Child (indexado para busca)

Subdivisão de um artigo — um parágrafo ou grupo de incisos.

**Texto do child:**
```
REN 1000/2021 — Estabelece os Procedimentos de Distribuição...
REN 1000/2021, Art. 7
---
§ 1º Microgeração distribuída: central geradora de energia elétrica, com potência
instalada menor ou igual a 75 kW e que utilize cogeração qualificada, conforme
regulamentação da ANEEL, ou fontes renováveis de energia elétrica, conectada na rede
de distribuição por meio de instalações de unidades consumidoras.
```

**Características:**
- 100–200 tokens (menor → embedding mais preciso)
- Embedado com bge-m3 (dense 1024-dim + sparse BM25)
- Payload inclui: `parent_id`, `texto_parent` (artigo completo), todos os metadados

---

## Lógica de Decisão

```
texto do PDF
    ↓
detecta "Art. Xº" via regex r'(?:^|\n)\s*(Art\.?\s*\d+[°º]?\.?)'?
    ├── SIM → chunk por artigo (parent) → divide em parágrafos/incisos (children)
    └── NÃO → RecursiveCharacterSplitter(size=500, overlap=80) → fallback
         ↓
sempre criar chunk de ementa separado (do JSON)
```

### Detecção de Artigos

```python
ARTICLE_RE = re.compile(
    r'(?:^|\n)\s*(Art\.?\s*\d+[°º]?[A-Z]?\.?\s*[-—]?\s*)',
    re.MULTILINE | re.IGNORECASE
)
```

Captura variações como:
- `Art. 1º` (mais comum)
- `Art. 1°` (grau em vez de ordinal)
- `Art. 10.` (ponto final)
- `ARTIGO 5` (por extenso)
- `Art. 1º-A` (artigo com letra)

### Divisão de Children

Hierarquia de divisão dentro de um artigo:

1. **Parágrafos** (`§ 1º`, `§ 2º`, ...) → cada parágrafo vira 1 child
2. **Incisos** (`I —`, `II —`, `III —`, ...) → agrupados em pares (2 incisos por child)
3. **Sem estrutura interna** → RecursiveCharacterTextSplitter(size=200, overlap=40)

### Fallback (sem artigos)

Usado principalmente para **Despachos (DSP)** — 53% do volume. São documentos curtos, sem estrutura de artigos, geralmente de conteúdo administrativo.

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,       # tokens
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", ", ", " "]
)
```

No fallback, o mesmo chunk é usado como parent e child (`parent_id = chunk_id`).

---

## Prefixo Contextual Universal

**Todo chunk começa com:**
```
{TIPO} {NUMERO}/{ANO} — {primeiros 150 chars da ementa}
[Art. N —] (se aplicável)
---
{texto do chunk}
```

**Por quê?** O modelo bge-m3 gera embeddings do chunk inteiro. Se o chunk for apenas `"§ 1º Microgeração distribuída: ..."`, sem contexto, o embedding não sabe que isso vem da REN 1000/2021. O prefixo garante que a identidade do documento seja carregada no vetor.

**Exemplo de chunk child com prefixo:**
```
REN 1000/2021 — Estabelece os Procedimentos de Distribuição de Energia Elétrica no Sistema Elétrico Nacional
REN 1000/2021, Art. 7
---
§ 1º Microgeração distribuída: central geradora de energia elétrica, com potência instalada
menor ou igual a 75 kW e que utilize cogeração qualificada...
```

---

## Chunking de Tabelas

Para documentos com tabelas Markdown (principalmente REH — Resoluções Homologatórias com tarifas):

```
| Subgrupo | Tarifa TE (R$/MWh) | Tarifa TUSD (R$/MWh) |
|----------|-------------------|---------------------|
| A1       | 123,45            | 67,89               |
| A2       | 134,56            | 72,10               |
...
```

**Estratégia:** `TABLE_ROWS_PER_CHUNK = 5` linhas por chunk, com o cabeçalho repetido em cada chunk:
```
[cabeçalho]    | Subgrupo | Tarifa TE | Tarifa TUSD |
[separador]    |----------|-----------|-------------|
[linhas 1-5]   | A1       | 123,45    | 67,89       |
               | A2       | 134,56    | 72,10       |
               ...
```

Isso garante que o LLM sempre saiba o que cada coluna significa, mesmo que leia apenas um chunk da tabela.

---

## Estatísticas do Dataset

| Tipo | Quantidade | % Total |
|------|-----------|---------|
| `child` | 111.531 | 48,9% |
| `fallback` | 89.757 | 39,4% |
| `ementa` | 26.768 | 11,7% |
| `table` | 22 | 0,01% |
| **Total** | **228.078** | **100%** |

**Indexados no Qdrant:** `child` + `ementa` + `fallback` + `table` = 228.078 pontos  
**Armazenados apenas no payload:** `texto_parent` de cada child

---

## Fluxo Completo — Exemplo Real

**Documento:** REN 1000/2021 (PRODIST)

```
PDF: ren20211000ti.pdf
    ↓
Parsing: texto limpo, 847 páginas, 9.823 artigos detectados
    ↓
Chunking:
    ├── 1 ementa (do JSON)             → chunk_id: ren20211000_e0000
    ├── Art. 1º (parent)               → armazenado no payload
    │   ├── "caput do Art. 1º"         → chunk_id: ren20211000_c0001
    │   ├── "§ 1º do Art. 1º"          → chunk_id: ren20211000_c0002
    │   └── "§ 2º do Art. 1º"          → chunk_id: ren20211000_c0003
    ├── Art. 2º (parent)               → armazenado no payload
    │   └── ...
    └── ...
```

**Ao receber a query "O que é microgeração distribuída?":**
1. bge-m3 embedda a query
2. Busca no Qdrant encontra `ren20211000_c0002` (§ 1º do Art. 7º)
3. Parent lookup: recupera `texto_parent` = Art. 7º completo (parágrafos § 1º ao § 8º)
4. LLM recebe o artigo completo → resposta contextualizada com citação `[REN 1000/2021, Art. 7º]`

---

## Comparação com Chunking Simples

**Chunking fixo (512 tokens, overlap 50):**
```
Chunk 1: "...Art. 6º O prazo para conexão [...] Art. 7º Para os fins desta Resolução,
          consideram-se: § 1º Microgeração distribuída: central geradora de energia
          elétrica, com potência instalada menor ou igual a 75 kW..."
          
Chunk 2: "...75 kW e que utilize cogeração qualificada, conforme regulamentação da ANEEL,
          ou fontes renováveis de energia elétrica, conectada na rede de distribuição por
          meio de instalações de unidades consumidoras. § 2º Minigeração distribuída:..."
```

❌ Art. 6 e Art. 7 misturados no Chunk 1  
❌ Definição de microgeração cortada entre Chunk 1 e Chunk 2  
❌ LLM pode alucinar completando a definição cortada  

**Parent-Child (nossa estratégia):**
```
Child indexado: "§ 1º Microgeração distribuída: central geradora..."
Parent recuperado: "Art. 7º Para os fins... § 1º Microgeração... § 2º Minigeração... § 3º..."
```

✅ Child pequeno → embedding preciso, busca certeira  
✅ Parent completo → LLM recebe o artigo inteiro, sem cortes  
✅ Citação correta → sempre `[REN 1000/2021, Art. 7º]`
