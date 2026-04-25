"""
Módulo 5.8 — Testes de embedding e indexação no Qdrant.

Os testes marcados com @pytest.mark.requires_index só passam APÓS rodar
05_embed_index.py (eles inspecionam o índice real em qdrant_db/).

Testes unitários (sem índice) rodam sempre.

Rode com:
    pytest tests/test_embed_index.py -v                        # todos
    pytest tests/test_embed_index.py -v -m "not requires_index"  # só unitários
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

QDRANT_PATH = "qdrant_db"
COLLECTION  = "aneel_legislacao"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_embed_module():
    spec = importlib.util.spec_from_file_location(
        "embed_index",
        Path(__file__).parent.parent / "src" / "05_embed_index.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def embed_mod():
    return _load_embed_module()


@pytest.fixture(scope="session")
def qdrant_client():
    """Abre o cliente Qdrant no índice real (requer índice criado)."""
    from qdrant_client import QdrantClient
    return QdrantClient(path=QDRANT_PATH)


def _collection_exists(client) -> bool:
    return any(c.name == COLLECTION for c in client.get_collections().collections)


# ── testes unitários (sem GPU, sem índice) ────────────────────────────────────

class TestChunkIdToInt:
    def test_deterministic(self, embed_mod):
        a = embed_mod.chunk_id_to_int("ren20211000_e0000")
        b = embed_mod.chunk_id_to_int("ren20211000_e0000")
        assert a == b

    def test_different_ids_give_different_ints(self, embed_mod):
        a = embed_mod.chunk_id_to_int("ren20211000_e0000")
        b = embed_mod.chunk_id_to_int("ren20211000_c0001")
        assert a != b

    def test_returns_positive_int(self, embed_mod):
        val = embed_mod.chunk_id_to_int("abc")
        assert isinstance(val, int) and val >= 0


class TestSparseDictToQdrant:
    def test_basic_conversion(self, embed_mod):
        sparse = embed_mod.sparse_dict_to_qdrant({"1": 0.5, "2": 0.3})
        assert set(sparse.indices) == {1, 2}
        assert len(sparse.values) == 2

    def test_empty_dict_returns_empty_vector(self, embed_mod):
        sparse = embed_mod.sparse_dict_to_qdrant({})
        assert sparse.indices == []
        assert sparse.values == []

    def test_string_keys_converted_to_int(self, embed_mod):
        sparse = embed_mod.sparse_dict_to_qdrant({"42": 0.9})
        assert sparse.indices[0] == 42
        assert isinstance(sparse.indices[0], int)


class TestBuildPoints:
    def test_point_has_correct_id(self, embed_mod):
        import numpy as np

        chunk = {"chunk_id": "ren20211000_e0000", "chunk_type": "ementa", "texto": "x"}
        embeddings = {
            "dense":  np.zeros((1, 1024)),
            "sparse": [{"1": 0.5}],
        }
        points = embed_mod.build_points([chunk], embeddings)
        assert len(points) == 1
        expected_id = embed_mod.chunk_id_to_int("ren20211000_e0000")
        assert points[0].id == expected_id

    def test_ementa_flag_set(self, embed_mod):
        import numpy as np

        chunk = {"chunk_id": "ren20211000_e0000", "chunk_type": "ementa", "texto": "x"}
        embeddings = {"dense": np.zeros((1, 1024)), "sparse": [{"1": 0.1}]}
        points = embed_mod.build_points([chunk], embeddings)
        assert points[0].payload["is_ementa"] is True

    def test_non_ementa_flag_false(self, embed_mod):
        import numpy as np

        chunk = {"chunk_id": "ren20211000_c0001", "chunk_type": "child", "texto": "x"}
        embeddings = {"dense": np.zeros((1, 1024)), "sparse": [{"1": 0.1}]}
        points = embed_mod.build_points([chunk], embeddings)
        assert points[0].payload["is_ementa"] is False

    def test_nan_payload_values_become_none(self, embed_mod):
        import numpy as np, math

        chunk = {"chunk_id": "x", "chunk_type": "child", "art_num": float("nan"), "texto": "x"}
        embeddings = {"dense": np.zeros((1, 1024)), "sparse": [{}]}
        points = embed_mod.build_points([chunk], embeddings)
        assert points[0].payload["art_num"] is None

    def test_dense_vector_length(self, embed_mod):
        import numpy as np

        chunk = {"chunk_id": "x", "chunk_type": "child", "texto": "x"}
        embeddings = {"dense": np.ones((1, 1024)), "sparse": [{"5": 0.2}]}
        points = embed_mod.build_points([chunk], embeddings)
        assert len(points[0].vector["dense"]) == 1024


class TestGetDevice:
    def test_returns_string(self, embed_mod):
        device = embed_mod.get_device()
        assert device in ("cuda", "mps", "cpu")


# ── testes de integração (requerem índice criado) ─────────────────────────────

@pytest.mark.requires_index
class TestQdrantIndex:
    def test_qdrant_collection_created(self, qdrant_client):
        assert _collection_exists(qdrant_client), (
            f"Collection '{COLLECTION}' não encontrada em '{QDRANT_PATH}'. "
            "Rode `python src/05_embed_index.py` primeiro."
        )

    def test_point_count_positive(self, qdrant_client):
        if not _collection_exists(qdrant_client):
            pytest.skip("Collection não existe.")
        count = qdrant_client.count(COLLECTION).count
        assert count > 0, "Índice está vazio."

    def test_dense_vector_dimension(self, qdrant_client):
        if not _collection_exists(qdrant_client):
            pytest.skip("Collection não existe.")
        results, _ = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=1,
            with_vectors=True,
        )
        assert results, "Nenhum ponto retornado."
        dense = results[0].vector.get("dense") if isinstance(results[0].vector, dict) else None
        assert dense is not None and len(dense) == 1024

    def test_sparse_vector_non_empty(self, qdrant_client):
        if not _collection_exists(qdrant_client):
            pytest.skip("Collection não existe.")
        results, _ = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=5,
            with_vectors=True,
        )
        for point in results:
            if not isinstance(point.vector, dict):
                continue
            sparse = point.vector.get("sparse")
            if sparse and hasattr(sparse, "indices") and len(sparse.indices) >= 1:
                return  # pelo menos 1 ponto com sparse não-vazio
        pytest.fail("Nenhum ponto encontrou sparse vector com índices.")

    def test_payload_has_texto_parent_for_children(self, qdrant_client):
        if not _collection_exists(qdrant_client):
            pytest.skip("Collection não existe.")
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results, _ = qdrant_client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_type", match=MatchValue(value="child"))]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            parent = point.payload.get("texto_parent")
            if parent and len(parent.strip()) > 0:
                return  # pelo menos 1 child com texto_parent
        pytest.fail("Nenhum chunk child com texto_parent encontrado.")

    def test_ementa_boost_field(self, qdrant_client):
        if not _collection_exists(qdrant_client):
            pytest.skip("Collection não existe.")
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results, _ = qdrant_client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_type", match=MatchValue(value="ementa"))]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False,
        )
        assert results, "Nenhum chunk de ementa encontrado."
        for point in results:
            assert point.payload.get("is_ementa") is True, (
                f"Ementa sem is_ementa=True: {point.payload.get('chunk_id')}"
            )

    def test_no_parent_chunks_indexed(self, qdrant_client):
        """Chunks do tipo 'parent' NÃO devem existir no índice."""
        if not _collection_exists(qdrant_client):
            pytest.skip("Collection não existe.")
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results, _ = qdrant_client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_type", match=MatchValue(value="parent"))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        assert not results, "Chunks do tipo 'parent' não deveriam estar no índice."
