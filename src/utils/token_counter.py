"""
Módulo 8.1 — Contagem e truncagem de tokens.
Ver roadmap.md para especificação completa.

TODO: implementar
"""
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Retorna o número de tokens no texto."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "cl100k_base") -> str:
    """Trunca o texto para no máximo max_tokens tokens."""
    enc = tiktoken.get_encoding(model)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
