"""
04_chunk.py
-----------
Gera chunks parent-child a partir dos textos parseados + metadados.

Estratégia (ver CLAUDE.md para detalhes completos):
1. Chunk de ementa — do JSON, 1 por documento, prioridade alta no retrieval
2. Chunks parent-child — document-aware:
   a. Detecta artigos via regex (Art. Xº, § Xº, incisos, alíneas)
   b. Parent = artigo completo (400-800 tokens) — NÃO indexado, fica no payload
   c. Child = parágrafo ou grupo de incisos (100-200 tokens) — indexado no Qdrant
   d. Fallback recursivo para docs sem estrutura (despachos, etc.)
3. Prefixo contextual em cada chunk: "{TIPO} {NUM}/{ANO} — {ementa[:150]}"

Entrada:  data/processed/texts/{doc_id}.txt
          data/processed/metadata.parquet
Saída:    data/processed/chunks.parquet
          data/processed/chunk_stats.json

Uso:
    python src/04_chunk.py
    python src/04_chunk.py --tipo REN REH REA
    python src/04_chunk.py --sample 100
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR
from src.utils.logger import logger
from src.utils.token_counter import count_tokens, truncate_to_tokens

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TEXTS_DIR         = PROCESSED_DIR / "texts"
METADATA_PATH     = PROCESSED_DIR / "metadata.parquet"
OUTPUT_PATH       = PROCESSED_DIR / "chunks.parquet"
STATS_PATH        = PROCESSED_DIR / "chunk_stats.json"

PARENT_MAX_TOKENS = 800
PARENT_MIN_TOKENS = 80
CHILD_MAX_TOKENS  = 200
CHILD_MIN_TOKENS  = 50
FALLBACK_SIZE     = 500
FALLBACK_OVERLAP  = 80
EMENTA_PREFIX_CHARS = 150
MAX_CHILDREN_PER_PARENT = 20
TABLE_ROWS_PER_CHUNK = 5        # linhas de dados por chunk de tabela

# Regex para estrutura jurídica (compilados uma vez)
ART_RE  = re.compile(
    r'(?:^|\n)([ \t]*Art\.?\s*\d+[°ºo]?(?:-[A-Z])?\s*[.:]?\s*)',
    re.MULTILINE,
)
PAR_RE  = re.compile(r'(?:^|\n)([ \t]*§\s*\d+[°ºo]?\.?\s*)', re.MULTILINE)
INC_RE  = re.compile(r'(?:^|\n)([ \t]*[IVXivx]+\s*[—–-]\s*)', re.MULTILINE)
ALIN_RE = re.compile(r'(?:^|\n)([ \t]*[a-z]\)\s*)', re.MULTILINE)

# Detecta blocos de tabela markdown (linhas com | em sequência)
TABLE_BLOCK_RE = re.compile(
    r'(?:^|\n)(\|[^\n]+\|(?:\n\|[^\n]+\|)+)',
    re.MULTILINE,
)


# ── 4.7 — Geração de IDs ──────────────────────────────────────────────────────

def generate_chunk_id(doc_id: str, chunk_type: str, seq: int) -> str:
    """Gera chunk_id determinístico: {doc_id}_{prefix}{seq:04d}.
    Prefixos: e=ementa, c=child, f=fallback, t=table
    """
    prefix = chunk_type[0]  # e=ementa, c=child, f=fallback, t=table
    return f"{doc_id}_{prefix}{seq:04d}"


def int_id_from_str(s: str) -> int:
    """MD5 hash → int para IDs do Qdrant."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**63)


# ── 4.6 — Prefixo contextual ──────────────────────────────────────────────────

def build_prefix(row: pd.Series, art_num: Optional[int] = None) -> str:
    """
    Gera prefixo contextual universal para todo chunk.
    Formato: '{TIPO} {NUM}/{ANO} — {ementa[:150]}\n[Art. N —]\n---'
    """
    tipo   = str(row.get("tipo_sigla", ""))
    numero = row.get("numero")
    ano    = row.get("ano", "")
    ementa = str(row.get("ementa", ""))[:EMENTA_PREFIX_CHARS]

    num_str = str(int(numero)) if pd.notna(numero) and numero else "s/n"
    header  = f"{tipo} {num_str}/{ano} — {ementa}"

    if art_num is not None:
        header += f"\nArt. {art_num}"

    return header + "\n---"


# ── 4.1 — Chunk de Ementa ─────────────────────────────────────────────────────

def build_ementa_chunk(row: pd.Series, seq: int = 0) -> dict:
    """
    Gera chunk de ementa a partir dos metadados do JSON.
    Criado para TODOS os documentos, independente de ter PDF.
    """
    tipo   = str(row.get("tipo_sigla", ""))
    tipo_nome = str(row.get("tipo_nome", ""))
    numero = row.get("numero")
    ano    = row.get("ano", "")
    num_str = str(int(numero)) if pd.notna(numero) and numero else "s/n"

    texto = (
        f"{tipo} {num_str}/{ano} — {row.get('ementa', '')}\n\n"
        f"Tipo: {tipo_nome}\n"
        f"Número: {num_str}\n"
        f"Ano: {ano}\n"
        f"Autor: {row.get('autor', '')}\n"
        f"Assunto: {row.get('assunto', '')}\n"
        f"Situação: {row.get('situacao', '')}\n"
        f"Data de publicação: {row.get('data_pub', '')}\n"
    )

    doc_id   = str(row["doc_id"])
    chunk_id = generate_chunk_id(doc_id, "ementa", seq)

    return {
        **_meta_fields(row),
        "chunk_id":    chunk_id,
        "chunk_type":  "ementa",
        "is_ementa":   True,
        "parent_id":   None,
        "texto_parent": None,
        "texto":       texto,
        "art_num":     None,
        "tokens":      count_tokens(texto),
    }


# ── 4.2 — Divisão por artigos ─────────────────────────────────────────────────

def split_by_articles(text: str) -> list[dict]:
    """
    Divide texto em artigos usando regex.
    Retorna lista de {art_num, art_text}.
    """
    matches = list(ART_RE.finditer(text))
    if not matches:
        return []

    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        art_text = text[start:end].strip()
        if not art_text:
            continue

        # extrair número do artigo
        num_match = re.search(r'\d+', m.group())
        art_num   = int(num_match.group()) if num_match else (i + 1)

        articles.append({"art_num": art_num, "art_text": art_text})

    return articles


# ── 4.3 + 4.4 — Parent e Children por artigo ─────────────────────────────────

def _split_into_children(art_text: str) -> list[str]:
    """
    Divide texto de um artigo em chunks child.
    Prioridade: parágrafos > incisos > alíneas > fallback por tokens.
    """
    # tentar dividir por parágrafos
    pars = PAR_RE.split(art_text)
    if len(pars) > 2:
        # PAR_RE.split retorna [antes, delim1, parte1, delim2, parte2, ...]
        segments = _join_split_parts(pars)
        if len(segments) > 1:
            return segments

    # tentar dividir por incisos
    incs = INC_RE.split(art_text)
    if len(incs) > 2:
        segments = _join_split_parts(incs)
        if len(segments) > 1:
            # agrupar em pares para não ter children muito curtos
            return _group_segments(segments, group_size=2)

    # tentar dividir por alíneas
    alins = ALIN_RE.split(art_text)
    if len(alins) > 2:
        segments = _join_split_parts(alins)
        if len(segments) > 1:
            return _group_segments(segments, group_size=3)

    # fallback: dividir por tokens
    return _split_by_tokens(art_text, CHILD_MAX_TOKENS, overlap=40)


def _join_split_parts(parts: list[str]) -> list[str]:
    """
    Reconstrói segmentos após re.split com grupo de captura.
    Formato de parts: [antes, delim, conteudo, delim, conteudo, ...]
    """
    result = []
    # primeiro elemento é o texto antes do primeiro delimitador
    if parts[0].strip():
        result.append(parts[0].strip())
    # depois vêm pares (delimitador, conteúdo)
    i = 1
    while i + 1 < len(parts):
        segment = (parts[i] + parts[i + 1]).strip()
        if segment:
            result.append(segment)
        i += 2
    return result


def _group_segments(segments: list[str], group_size: int) -> list[str]:
    """Agrupa segmentos de N em N."""
    groups = []
    for i in range(0, len(segments), group_size):
        group = "\n".join(s for s in segments[i:i + group_size] if s)
        if group:
            groups.append(group)
    return groups


def _split_by_tokens(text: str, max_tokens: int, overlap: int = 0) -> list[str]:
    """Divide texto em chunks de max_tokens com overlap."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start  = 0
    while start < len(words):
        end   = start
        accum = 0
        while end < len(words):
            token_count = count_tokens(words[end])
            if accum + token_count > max_tokens and end > start:
                break
            accum += token_count
            end   += 1

        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(words):
            break
        # calcular overlap em palavras
        overlap_words = max(1, overlap // 4)
        start = max(start + 1, end - overlap_words)

    return chunks if chunks else [text]


def build_parent_and_children(
    row: pd.Series,
    art_num: int,
    art_text: str,
    seq_start: int,
) -> tuple[dict, list[dict]]:
    """
    Gera parent + lista de children para um artigo.
    Parent NÃO é indexado — fica no campo texto_parent dos children.
    Retorna (parent_dict, [child_dict, ...]).
    """
    doc_id  = str(row["doc_id"])
    prefix  = build_prefix(row, art_num)
    parent_text = f"{prefix}\n{art_text}"

    # truncar parent se muito longo
    if count_tokens(parent_text) > PARENT_MAX_TOKENS:
        parent_text = truncate_to_tokens(parent_text, PARENT_MAX_TOKENS)

    parent_id = f"{doc_id}_art{art_num:03d}"
    parent = {
        **_meta_fields(row),
        "chunk_id":    parent_id,
        "chunk_type":  "parent",
        "is_ementa":   False,
        "parent_id":   None,
        "texto_parent": None,
        "texto":       parent_text,
        "art_num":     art_num,
        "tokens":      count_tokens(parent_text),
    }

    # gerar children
    child_texts = _split_into_children(art_text)
    # limitar número de children
    if len(child_texts) > MAX_CHILDREN_PER_PARENT:
        child_texts = _group_segments(child_texts, group_size=3)
        child_texts = child_texts[:MAX_CHILDREN_PER_PARENT]

    children = []
    for i, child_text in enumerate(child_texts):
        if not child_text.strip():
            continue
        child_full = f"{prefix}\n{child_text}"
        # truncar child se necessário
        if count_tokens(child_full) > CHILD_MAX_TOKENS:
            child_full = truncate_to_tokens(child_full, CHILD_MAX_TOKENS)

        chunk_id = generate_chunk_id(doc_id, "child", seq_start + i)
        children.append({
            **_meta_fields(row),
            "chunk_id":    chunk_id,
            "chunk_type":  "child",
            "is_ementa":   False,
            "parent_id":   parent_id,
            "texto_parent": parent_text,
            "texto":       child_full,
            "art_num":     art_num,
            "tokens":      count_tokens(child_full),
        })

    # se não gerou nenhum child, o artigo inteiro vira um child
    if not children:
        chunk_id = generate_chunk_id(doc_id, "child", seq_start)
        children.append({
            **_meta_fields(row),
            "chunk_id":    chunk_id,
            "chunk_type":  "child",
            "is_ementa":   False,
            "parent_id":   parent_id,
            "texto_parent": parent_text,
            "texto":       f"{prefix}\n{art_text}",
            "art_num":     art_num,
            "tokens":      count_tokens(f"{prefix}\n{art_text}"),
        })

    return parent, children


# ── 4.5 — Fallback para docs sem artigos ──────────────────────────────────────

def build_fallback_chunks(row: pd.Series, full_text: str) -> list[dict]:
    """
    Chunking recursivo por tamanho para documentos sem estrutura de artigos.
    Cada bloco é simultaneamente parent e child (chunk_type='fallback').
    """
    doc_id = str(row["doc_id"])
    prefix = build_prefix(row)

    segments = _split_by_tokens(full_text, FALLBACK_SIZE, overlap=FALLBACK_OVERLAP)
    chunks   = []
    for i, seg in enumerate(segments):
        if not seg.strip():
            continue
        texto    = f"{prefix}\n{seg}"
        chunk_id = generate_chunk_id(doc_id, "fallback", i)
        chunks.append({
            **_meta_fields(row),
            "chunk_id":    chunk_id,
            "chunk_type":  "fallback",
            "is_ementa":   False,
            "parent_id":   chunk_id,       # aponta para si mesmo
            "texto_parent": texto,
            "texto":       texto,
            "art_num":     None,
            "tokens":      count_tokens(texto),
        })

    return chunks


# ── 4.8a — Chunking de tabelas (linha descritiva + linha informativa) ─────────

def extract_markdown_tables(text: str) -> list[dict]:
    """
    Extrai blocos de tabela markdown do texto.
    Retorna lista de {header, separator, rows, context_before}.

    Cada bloco tem:
      - header:        linha de cabeçalho (linha descritiva das colunas)
      - separator:     linha de traços |---|---|
      - rows:          lista de strings de linhas de dados (linhas informativas)
      - context_before: até 200 chars antes da tabela (título, artigo, etc.)
    """
    tables = []
    for m in TABLE_BLOCK_RE.finditer(text):
        block = m.group(1).strip()
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue

        header    = lines[0]
        # separator é opcional (linha com |---|)
        sep_idx = 1 if (len(lines) > 1 and re.match(r'^\|[-| :]+\|$', lines[1].strip())) else None
        if sep_idx is not None:
            separator = lines[sep_idx]
            data_rows = lines[sep_idx + 1:]
        else:
            separator = ""
            data_rows = lines[1:]

        if not data_rows:
            continue

        # contexto antes da tabela (título/artigo)
        start_pos = m.start()
        context_before = text[max(0, start_pos - 200):start_pos].strip()
        # pegar só a última linha do contexto (mais relevante)
        context_lines = [ln.strip() for ln in context_before.splitlines() if ln.strip()]
        context = context_lines[-1] if context_lines else ""

        tables.append({
            "header":         header,
            "separator":      separator,
            "rows":           data_rows,
            "context_before": context,
        })

    return tables


def build_table_chunks(
    row: pd.Series,
    tables: list[dict],
    seq_start: int = 0,
) -> list[dict]:
    """
    Gera chunks de tabela com a técnica linha-descritiva + linha-informativa.

    Cada chunk contém:
      - prefixo contextual do documento
      - contexto (título/artigo antes da tabela)
      - linha de cabeçalho (linha descritiva) repetida em TODOS os chunks
      - N linhas de dados (linhas informativas)

    chunk_type = 'table'
    parent_id  = chunk_id (auto-referência, como fallback)
    """
    doc_id = str(row["doc_id"])
    prefix = build_prefix(row)
    chunks: list[dict] = []
    seq = seq_start

    for tbl in tables:
        header    = tbl["header"]
        separator = tbl["separator"]
        rows      = tbl["rows"]
        context   = tbl["context_before"]

        # Linha descritiva: header + separator (se existir)
        desc_line = header
        if separator:
            desc_line = f"{header}\n{separator}"

        # Dividir linhas informativas em grupos de TABLE_ROWS_PER_CHUNK
        for i in range(0, max(1, len(rows)), TABLE_ROWS_PER_CHUNK):
            batch = rows[i : i + TABLE_ROWS_PER_CHUNK]
            if not batch:
                continue

            # Montar texto do chunk
            parts = [prefix]
            if context:
                parts.append(context)
            parts.append(desc_line)
            parts.extend(batch)
            texto = "\n".join(parts)

            chunk_id = generate_chunk_id(doc_id, "table", seq)
            chunks.append({
                **_meta_fields(row),
                "chunk_id":    chunk_id,
                "chunk_type":  "table",
                "is_ementa":   False,
                "parent_id":   chunk_id,   # auto-referência
                "texto_parent": texto,
                "texto":       texto,
                "art_num":     None,
                "tokens":      count_tokens(texto),
            })
            seq += 1

    return chunks


# ── 4.8 — Pipeline completo por documento ─────────────────────────────────────

def chunk_document(row: pd.Series, full_text: str) -> list[dict]:
    """
    Gera todos os chunks de um documento:
    1. Ementa (sempre)
    2. Chunks de tabela (linha descritiva + linhas informativas), se houver tabelas markdown
    3. Parent+children por artigo (se houver)  OU  fallback (se não)
    Retorna lista de dicts prontos para o Qdrant.
    """
    chunks: list[dict] = []

    # 1. Ementa — sempre criada
    chunks.append(build_ementa_chunk(row, seq=0))

    if not full_text or not full_text.strip():
        return chunks

    # 2. Chunks de tabela (independentes da estrutura de artigos)
    tables = extract_markdown_tables(full_text)
    if tables:
        table_chunks = build_table_chunks(row, tables, seq_start=0)
        chunks.extend(table_chunks)

    # 3. Detectar artigos
    articles = split_by_articles(full_text)

    if articles:
        seq = 0
        for art in articles:
            parent, children = build_parent_and_children(
                row      = row,
                art_num  = art["art_num"],
                art_text = art["art_text"],
                seq_start= seq,
            )
            # parent NÃO vai para o Qdrant — apenas armazenado nos children
            chunks.extend(children)
            seq += len(children)
    else:
        # fallback (apenas se não houver artigos; tabelas já foram tratadas acima)
        chunks.extend(build_fallback_chunks(row, full_text))

    return chunks


# ── helpers ───────────────────────────────────────────────────────────────────

def _meta_fields(row: pd.Series) -> dict:
    """Extrai campos de metadados comuns a todos os chunks."""
    return {
        "doc_id":    str(row["doc_id"]),
        "tipo_sigla": str(row.get("tipo_sigla", "")),
        "tipo_nome":  str(row.get("tipo_nome", "")),
        "numero":     row.get("numero"),
        "ano":        row.get("ano"),
        "data_pub":   str(row.get("data_pub", "") or ""),
        "data_ass":   str(row.get("data_ass", "") or ""),
        "autor":      str(row.get("autor", "") or ""),
        "assunto":    str(row.get("assunto", "") or ""),
        "ementa":     str(row.get("ementa", "") or ""),
        "situacao":   str(row.get("situacao", "") or ""),
        "revogada":   bool(row.get("revogada", False)),
        "pdf_url":    str(row.get("pdf_url", "") or ""),
    }


# ── 4.9 — Pipeline batch ──────────────────────────────────────────────────────

def run_chunk(
    tipo_filter: Optional[list[str]] = None,
    sample: Optional[int] = None,
) -> pd.DataFrame:
    """
    Processa todos os documentos e salva chunks.parquet.
    """
    if not METADATA_PATH.exists():
        logger.error(f"metadata.parquet não encontrado. Rode 01_consolidate_metadata.py primeiro.")
        return pd.DataFrame()

    df_meta = pd.read_parquet(METADATA_PATH)
    logger.info(f"Metadados carregados: {len(df_meta)} documentos")

    if tipo_filter:
        tipos = [t.upper() for t in tipo_filter]
        df_meta = df_meta[df_meta["tipo_sigla"].isin(tipos)]
        logger.info(f"Filtro tipo={tipos}: {len(df_meta)} documentos")

    if sample:
        df_meta = df_meta.sample(min(sample, len(df_meta)), random_state=42)
        logger.info(f"Amostra: {len(df_meta)} documentos")

    all_chunks: list[dict] = []
    sem_texto  = 0

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Chunking"):
        doc_id   = str(row["doc_id"])
        txt_path = TEXTS_DIR / f"{doc_id}.txt"

        if txt_path.exists():
            full_text = txt_path.read_text(encoding="utf-8")
        else:
            full_text = ""
            sem_texto += 1

        chunks = chunk_document(row, full_text)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("Nenhum chunk gerado.")
        return pd.DataFrame()

    df = pd.DataFrame(all_chunks)

    # garantir unicidade de chunk_id
    dupes = df["chunk_id"].duplicated().sum()
    if dupes:
        logger.warning(f"{dupes} chunk_ids duplicados — adicionando sufixo hash")
        df["chunk_id"] = df["chunk_id"] + "_" + df.groupby("chunk_id").cumcount().astype(str)

    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"chunks.parquet salvo: {len(df)} chunks")

    if sem_texto:
        logger.warning(f"{sem_texto} documentos sem .txt (só ementa gerada)")

    _save_stats(df)
    return df


def _save_stats(df: pd.DataFrame) -> None:
    stats = {
        "total_chunks":   len(df),
        "por_tipo":       df["chunk_type"].value_counts().to_dict(),
        "por_sigla":      df["tipo_sigla"].value_counts().to_dict(),
        "token_stats": {
            "mean":   round(df["tokens"].mean(), 1),
            "median": round(df["tokens"].median(), 1),
            "p95":    round(df["tokens"].quantile(0.95), 1),
            "max":    int(df["tokens"].max()),
        },
        "docs_unicos":    df["doc_id"].nunique(),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(f"Stats: {stats}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Módulo 4 — Chunking Document-Aware")
    parser.add_argument("--tipo",   nargs="+", help="Filtrar por tipo (ex: REN REH)")
    parser.add_argument("--sample", type=int,  help="Processar apenas N documentos")
    args = parser.parse_args()

    df = run_chunk(tipo_filter=args.tipo, sample=args.sample)
    if not df.empty:
        print(df[["doc_id", "tipo_sigla", "chunk_type", "art_num", "tokens"]].head(15).to_string())
