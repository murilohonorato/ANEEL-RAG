"""
04_chunk.py
-----------
Gera chunks parent-child a partir dos textos parseados + metadados.

Estratégia (ver CLAUDE.md para detalhes completos):
1. Chunk de ementa — do JSON, 1 por documento, prioridade alta
2. Chunks parent-child — document-aware:
   a. Detecta artigos via regex (Art. Xº, § Xº, incisos, alíneas)
   b. Parent = artigo completo (400-800 tokens)
   c. Child = parágrafo ou grupo de incisos (100-200 tokens)
   d. Fallback recursivo para docs sem estrutura (despachos curtos)
3. Prefixo contextual em cada chunk: "{TIPO} {NUM}/{ANO} — {ementa[:150]}"

Entrada:  data/processed/texts/{doc_id}.txt
          data/processed/metadata.parquet
Saída:    data/processed/chunks.parquet

Schema do chunk:
    doc_id, tipo_sigla, tipo_nome, numero, ano, data_pub, data_ass,
    autor, assunto, ementa, situacao, pdf_url,
    chunk_id, chunk_type (ementa|child|fallback),
    parent_id, texto_parent, texto, art_num, tokens

Uso:
    python src/04_chunk.py
    python src/04_chunk.py --tipo REN REH REA  # filtrar tipos
    python src/04_chunk.py --sample 100        # testar em 100 docs
"""

# CONFIG
TEXTS_DIR    = "data/processed/texts"
METADATA_PATH = "data/processed/metadata.parquet"
OUTPUT_PATH  = "data/processed/chunks.parquet"

PARENT_MAX_TOKENS = 800
PARENT_MIN_TOKENS = 80
CHILD_MAX_TOKENS  = 200
CHILD_MIN_TOKENS  = 50
FALLBACK_SIZE     = 500
FALLBACK_OVERLAP  = 80

# Regex para detectar estrutura jurídica
import re
ART_PATTERN = re.compile(r'^\s*Art\.?\s*\d+[ºo°]?\.?\s', re.MULTILINE)
PAR_PATTERN = re.compile(r'^\s*§\s*\d+[ºo°]?\.?\s', re.MULTILINE)

# TODO: implementar com Claude Code
