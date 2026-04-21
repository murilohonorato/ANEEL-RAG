"""
Módulo 8.3 — Construção de filtros para o Qdrant.
Ver roadmap.md para especificação completa.

TODO: implementar
"""
from typing import Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range


def build_filter(filters: dict) -> Optional[Filter]:
    """
    Constrói um Filter do Qdrant a partir de um dict de filtros.

    Chaves suportadas: ano (int), tipo_sigla (str), revogada (bool), chunk_type (str)

    Exemplo:
        build_filter({"ano": 2021, "tipo_sigla": "REN"})
    """
    if not filters:
        return None

    conditions = []

    if "ano" in filters and filters["ano"]:
        conditions.append(
            FieldCondition(key="ano", match=MatchValue(value=int(filters["ano"])))
        )
    if "tipo_sigla" in filters and filters["tipo_sigla"]:
        conditions.append(
            FieldCondition(key="tipo_sigla", match=MatchValue(value=filters["tipo_sigla"]))
        )
    if "revogada" in filters and filters["revogada"] is not None:
        conditions.append(
            FieldCondition(key="revogada", match=MatchValue(value=bool(filters["revogada"])))
        )
    if "chunk_type" in filters and filters["chunk_type"]:
        conditions.append(
            FieldCondition(key="chunk_type", match=MatchValue(value=filters["chunk_type"]))
        )

    if not conditions:
        return None

    return Filter(must=conditions)
