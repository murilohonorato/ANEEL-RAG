"""
05_embed_index.py
-----------------
Embeda os chunks e indexa no Qdrant.

- Modelo: BAAI/bge-m3 (dense 1024d + sparse BM25-like)
- Indexa: chunks do tipo "child", "ementa" e "fallback"
- NÃO indexa: chunks do tipo "parent" (ficam só no payload dos children)
- Armazena texto_parent no payload para parent lookup no módulo 6
- Ementas marcadas com is_ementa=True para boost na query
- Suporte a GPU (CUDA) com fp16 para economizar VRAM

Entrada:  data/processed/chunks.parquet
Saída:    qdrant_db/  (índice vetorial em disco)
          data/processed/index_stats.json

Uso:
    python src/05_embed_index.py
    python src/05_embed_index.py --reset          # recria o índice do zero
    python src/05_embed_index.py --batch-size 8   # reduzir se OOM
    python src/05_embed_index.py --incremental    # pula doc_ids já indexados
    python src/05_embed_index.py --dry-run        # mostra estatísticas sem indexar
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CHUNKS_PATH  = "data/processed/chunks.parquet"
QDRANT_PATH  = "qdrant_db"
COLLECTION   = "aneel_legislacao"
EMBED_MODEL  = "BAAI/bge-m3"

# Batch size de embedding — limitado pela VRAM.
# GTX 1660 6 GB: usa ~2.7 GB com batch=16; batch=32 usa ~3.5 GB (seguro).
# Se aparecer OOM, reduza para 16. Com VRAM >= 10 GB, pode tentar 64.
BATCH_SIZE    = 32

# Batch size de upsert — quantos pontos são enviados ao Qdrant por chamada.
# Valores maiores reduzem o overhead de I/O do modo local.
# Não afeta VRAM — os pontos ficam em RAM até o flush.
UPSERT_BATCH  = 512

DENSE_DIM    = 1024   # dimensão do bge-m3
HNSW_M       = 16
HNSW_EF      = 200

# Tipos que vão para o índice vetorial
INDEXABLE_TYPES = {"child", "ementa", "fallback"}


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    """Detecta automaticamente: CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detectada: {name} ({vram_gb:.1f} GB VRAM)")
            return "cuda"
        if torch.backends.mps.is_available():
            logger.info("Apple MPS detectado.")
            return "mps"
    except ImportError:
        pass
    logger.warning("Nenhuma GPU detectada — usando CPU (será lento).")
    return "cpu"


# ── Modelo ────────────────────────────────────────────────────────────────────

def load_model(device: str):
    """Carrega BAAI/bge-m3. Baixa automaticamente na primeira execução (~2 GB)."""
    from FlagEmbedding import BGEM3FlagModel

    use_fp16 = device in ("cuda", "mps")
    logger.info(f"Carregando {EMBED_MODEL} (fp16={use_fp16}, device={device})…")
    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=use_fp16, device=device)
    logger.info("Modelo carregado.")
    return model


# ── Qdrant ────────────────────────────────────────────────────────────────────

def get_qdrant_client(path: str):
    from qdrant_client import QdrantClient
    return QdrantClient(path=path)


def create_collection(client, name: str) -> None:
    """Cria a collection com vetores densos + esparsos e índices de payload."""
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        OptimizersConfigDiff,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            ),
        },
        hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF),
        optimizers_config=OptimizersConfigDiff(memmap_threshold=20_000),
    )
    logger.info(f"Collection '{name}' criada.")

    # Índices de payload para filtros rápidos
    from qdrant_client.models import PayloadSchemaType
    index_fields = {
        "tipo_sigla":  PayloadSchemaType.KEYWORD,
        "ano":         PayloadSchemaType.INTEGER,
        "chunk_type":  PayloadSchemaType.KEYWORD,
        "is_ementa":   PayloadSchemaType.BOOL,
        "revogada":    PayloadSchemaType.BOOL,
    }
    for field, schema in index_fields.items():
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
        )
    logger.info("Índices de payload criados.")


def get_or_create_collection(client, name: str, reset: bool) -> None:
    """Garante que a collection existe. Se reset=True, recria do zero."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    existing = {c.name for c in client.get_collections().collections}

    if reset and name in existing:
        client.delete_collection(name)
        logger.info(f"Collection '{name}' deletada para reset.")
        existing.discard(name)

    if name not in existing:
        create_collection(client, name)
    else:
        info  = client.get_collection(name)
        count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
        logger.info(f"Collection '{name}' já existe com {count:,} vetores.")


def get_indexed_doc_ids(client, name: str) -> set[str]:
    """Recupera doc_ids já presentes na collection (para modo incremental)."""
    from qdrant_client.models import ScrollRequest

    indexed: set[str] = set()
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        for point in result:
            if point.payload and "doc_id" in point.payload:
                indexed.add(point.payload["doc_id"])
        if next_offset is None:
            break
        offset = next_offset

    return indexed


# ── IDs ───────────────────────────────────────────────────────────────────────

def chunk_id_to_int(chunk_id: str) -> int:
    """Converte chunk_id string em int estável via MD5 (Qdrant exige int ou UUID)."""
    digest = hashlib.md5(chunk_id.encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], model, batch_size: int) -> dict:
    """
    Gera embeddings densos e esparsos para uma lista de textos.

    Retorna:
        {
            "dense":  ndarray[N, 1024],
            "sparse": list[dict{int → float}]  ← um dict por texto
        }
    """
    all_dense  = []
    all_sparse = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch", leave=False):
        batch = texts[i : i + batch_size]
        out = model.encode(
            batch,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        all_dense.append(out["dense_vecs"])
        # sparse: lista de dicts {token_id_str: weight}
        all_sparse.extend(out["lexical_weights"])

    import numpy as np
    return {
        "dense":  np.concatenate(all_dense, axis=0),
        "sparse": all_sparse,
    }


def sparse_dict_to_qdrant(sparse_dict: dict):
    """Converte {token_id: weight} do bge-m3 para SparseVector do Qdrant."""
    from qdrant_client.models import SparseVector

    if not sparse_dict:
        # vetor esparso vazio — chunk sem tokens reconhecidos (raro)
        return SparseVector(indices=[], values=[])

    indices = [int(k) for k in sparse_dict]
    values  = [float(v) for v in sparse_dict.values()]
    return SparseVector(indices=indices, values=values)


# ── Montagem dos pontos ───────────────────────────────────────────────────────

# Campos que vão para o payload (excluímos chunk_id pq é duplicado como campo)
_PAYLOAD_EXCLUDE = set()

def build_points(chunks: list[dict], embeddings: dict) -> list:
    """Cria lista de PointStruct pronta para upsert no Qdrant."""
    from qdrant_client.models import PointStruct

    points = []
    dense_vecs  = embeddings["dense"]
    sparse_vecs = embeddings["sparse"]

    for i, chunk in enumerate(chunks):
        chunk_id  = chunk.get("chunk_id", "")
        point_id  = chunk_id_to_int(chunk_id)

        payload = {k: v for k, v in chunk.items()}
        # Garante flag de ementa no payload
        payload["is_ementa"] = (chunk.get("chunk_type") == "ementa")

        # NaN do pandas → None (Qdrant não aceita float NaN)
        for key, val in payload.items():
            if isinstance(val, float) and (val != val):  # NaN check sem numpy
                payload[key] = None

        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense":  dense_vecs[i].tolist(),
                    "sparse": sparse_dict_to_qdrant(sparse_vecs[i]),
                },
                payload=payload,
            )
        )
    return points


# ── Upload com retry ──────────────────────────────────────────────────────────

def upload_batch(client, collection: str, points: list, retries: int = 3) -> None:
    """Faz upsert de um lote de pontos com retry exponencial."""
    delay = 1.0
    for attempt in range(retries):
        try:
            client.upsert(collection_name=collection, points=points)
            return
        except Exception as exc:
            if attempt == retries - 1:
                raise
            logger.warning(f"Upsert falhou ({exc}), tentativa {attempt + 2}/{retries} em {delay:.0f}s…")
            time.sleep(delay)
            delay *= 2


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(
    chunks_path: str,
    qdrant_path: str,
    collection: str,
    batch_size: int,
    reset: bool,
    incremental: bool,
    dry_run: bool,
    upsert_batch: int = UPSERT_BATCH,
) -> None:
    # 1. Carregar chunks
    logger.info(f"Carregando chunks de {chunks_path}…")
    df = pd.read_parquet(chunks_path)
    logger.info(f"Total de chunks no parquet: {len(df):,}")
    logger.info(f"Tipos:\n{df['chunk_type'].value_counts().to_string()}")

    # 2. Filtrar indexáveis
    df_index = df[df["chunk_type"].isin(INDEXABLE_TYPES)].copy()
    logger.info(f"Chunks a indexar (child+ementa+fallback): {len(df_index):,}")

    # 3. Remover chunks com texto vazio (PDFs não disponíveis, 98 documentos)
    missing_mask = df_index["texto"].isna() | (df_index["texto"].str.strip() == "")
    n_missing = missing_mask.sum()
    if n_missing:
        missing_docs = df_index.loc[missing_mask, "doc_id"].unique().tolist()
        logger.warning(
            f"{n_missing} chunks sem texto serão ignorados "
            f"({len(missing_docs)} doc_ids — PDFs não disponíveis): "
            f"{missing_docs[:10]}{'…' if len(missing_docs) > 10 else ''}"
        )
        df_index = df_index[~missing_mask]
    logger.info(f"Chunks com texto válido: {len(df_index):,}")

    if dry_run:
        logger.info("[DRY-RUN] Estatísticas calculadas. Nenhum dado foi indexado.")
        return

    # 4. Configurar Qdrant
    client = get_qdrant_client(qdrant_path)
    get_or_create_collection(client, collection, reset)

    # 5. Modo incremental — pular doc_ids já indexados
    if incremental and not reset:
        already_indexed = get_indexed_doc_ids(client, collection)
        before = len(df_index)
        df_index = df_index[~df_index["doc_id"].isin(already_indexed)]
        logger.info(
            f"Modo incremental: {before - len(df_index):,} chunks já indexados ignorados. "
            f"Restam: {len(df_index):,}"
        )

    if df_index.empty:
        logger.info("Nenhum chunk novo para indexar. Finalizando.")
        _save_stats(client, collection, qdrant_path)
        return

    # 6. Liberar memória antes de carregar o modelo
    # O client Qdrant local pode manter dados em RAM durante o scroll incremental.
    # Fechar e reabrir depois garante que o modelo carrega sem pressão de memória.
    del client
    import gc
    gc.collect()
    logger.info("Memória liberada. Carregando modelo...")

    device = get_device()
    model  = load_model(device)

    # Reabrir client após modelo carregado
    client = get_qdrant_client(qdrant_path)

    # 7. Processar em batches
    records    = df_index.to_dict("records")
    texts      = [r["texto"] for r in records]
    total      = len(records)
    n_uploaded = 0

    logger.info(
        f"Iniciando embedding e indexação de {total:,} chunks "
        f"(embed_batch={batch_size}, upsert_batch={upsert_batch})…"
    )
    start_time = time.time()

    pending_points: list = []
    n_processed   = 0
    LOG_EVERY     = 500  # loga progresso a cada N chunks processados

    def _flush(force: bool = False) -> None:
        nonlocal n_uploaded
        if not pending_points:
            return
        if not force and len(pending_points) < upsert_batch:
            return
        try:
            upload_batch(client, collection, pending_points)
            n_uploaded += len(pending_points)
        except Exception as exc:
            logger.error(f"Erro ao fazer upsert de {len(pending_points)} pontos: {exc}")
        pending_points.clear()

    def _log_progress() -> None:
        if n_processed == 0:
            return
        elapsed   = time.time() - start_time
        rate      = n_processed / elapsed          # chunks/s
        remaining = total - n_processed
        eta_s     = remaining / rate if rate > 0 else 0
        eta_h     = int(eta_s // 3600)
        eta_m     = int((eta_s % 3600) // 60)
        pct       = n_processed / total * 100
        logger.info(
            f"Progresso: {n_processed:,}/{total:,} chunks ({pct:.1f}%) | "
            f"{rate * 60:.0f} chunks/min | "
            f"ETA: {eta_h}h {eta_m:02d}min"
        )

    for i in tqdm(range(0, total, batch_size), desc="Indexando", unit="batch"):
        chunk_batch = records[i : i + batch_size]
        text_batch  = texts[i : i + batch_size]

        try:
            embeddings = embed_texts(text_batch, model, batch_size)
        except Exception as exc:
            logger.error(f"Erro ao embedar batch {i}–{i+batch_size}: {exc}")
            continue

        pending_points.extend(build_points(chunk_batch, embeddings))
        n_processed += len(chunk_batch)
        _flush()

        if n_processed % LOG_EVERY < batch_size:
            _log_progress()

    _flush(force=True)  # envia o restante
    _log_progress()     # progresso final

    elapsed = time.time() - start_time
    logger.info(f"Concluído: {n_uploaded:,} pontos indexados em {elapsed/60:.1f} min.")

    # 8. Otimizar segmentos
    logger.info("Otimizando segmentos do Qdrant…")
    client.update_collection(
        collection_name=collection,
        optimizer_config={"indexing_threshold": 0},
    )

    # 9. Salvar estatísticas
    _save_stats(client, collection, qdrant_path, elapsed_seconds=elapsed)


def _save_stats(client, collection: str, qdrant_path: str, elapsed_seconds: float = 0.0) -> None:
    """Salva data/processed/index_stats.json com métricas de indexação."""
    info = client.get_collection(collection)
    vectors_count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0

    stats_path = Path("data/processed/index_stats.json")

    # Ler stats existentes se houver (para preservar chunk_stats do módulo 4)
    existing: dict = {}
    if stats_path.exists():
        try:
            existing = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    existing.update({
        "indexed_chunks":         vectors_count,
        "collection":             collection,
        "qdrant_path":            str(qdrant_path),
        "indexing_time_minutes":  round(elapsed_seconds / 60, 2),
    })

    stats_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Estatísticas salvas em {stats_path} ({vectors_count:,} vetores).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Módulo 5 — Embedding e indexação no Qdrant.")
    p.add_argument("--chunks",      default=CHUNKS_PATH,  help="Caminho do chunks.parquet")
    p.add_argument("--qdrant",      default=QDRANT_PATH,  help="Diretório do Qdrant")
    p.add_argument("--collection",  default=COLLECTION,   help="Nome da collection")
    p.add_argument("--batch-size",  default=BATCH_SIZE,   type=int, help="Embedding batch size (padrão: 32)")
    p.add_argument("--upsert-batch", default=UPSERT_BATCH, type=int, help="Upsert batch size (padrão: 512)")
    p.add_argument("--reset",       action="store_true",  help="Deletar e recriar a collection")
    p.add_argument("--incremental", action="store_true",  help="Pular doc_ids já indexados")
    p.add_argument("--dry-run",     action="store_true",  help="Mostrar stats sem indexar")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        chunks_path  = args.chunks,
        qdrant_path  = args.qdrant,
        collection   = args.collection,
        batch_size   = args.batch_size,
        upsert_batch = args.upsert_batch,
        reset        = args.reset,
        incremental  = args.incremental,
        dry_run      = args.dry_run,
    )
