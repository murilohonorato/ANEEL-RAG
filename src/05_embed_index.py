"""
05_embed_index.py
-----------------
Embeda os chunks e indexa no Qdrant.

- Modelo: BAAI/bge-m3 (dense 1024d + sparse)
- Indexa: children + ementas
- Armazena no payload: texto_parent completo (para parent lookup)
- Ementas com score boost via named vector ou campo dedicado
- Processa em batches para eficiência de memória/GPU

Entrada:  data/processed/chunks.parquet
Saída:    qdrant_db/ (índice em disco)
          data/processed/index_stats.json

Uso:
    python src/05_embed_index.py
    python src/05_embed_index.py --reset   # recria o índice do zero
    python src/05_embed_index.py --batch-size 64
"""

# CONFIG
CHUNKS_PATH    = "data/processed/chunks.parquet"
QDRANT_PATH    = "qdrant_db"
COLLECTION     = "aneel_legislacao"
EMBED_MODEL    = "BAAI/bge-m3"
BATCH_SIZE     = 32
EMENTA_BOOST   = 1.5    # multiplicador de score para chunks de ementa

# Parâmetros HNSW
HNSW_M         = 16
HNSW_EF        = 200

# TODO: implementar com Claude Code
