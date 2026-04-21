"""
Testes para o módulo de chunking.
Rode com: pytest tests/test_chunk.py -v
"""

# TODO: implementar com Claude Code
# Casos de teste esperados:
#
# 1. test_ementa_chunk_always_created
#    - dado qualquer registro do JSON
#    - deve sempre gerar exatamente 1 chunk de ementa
#    - com campos: tipo_sigla, numero, ano, ementa, chunk_type="ementa"
#
# 2. test_artigo_detection
#    - dado texto com "Art. 1º ... Art. 2º ..."
#    - deve detectar 2 artigos e criar 2 parents
#
# 3. test_parent_child_link
#    - dado um artigo com 3 parágrafos
#    - children devem ter parent_id apontando para o mesmo parent
#    - parent deve estar no payload de cada child
#
# 4. test_fallback_for_despacho
#    - dado texto curto sem estrutura de artigos (despacho típico)
#    - deve usar fallback recursivo
#    - chunk_type deve ser "fallback"
#
# 5. test_contextual_prefix
#    - todo chunk deve começar com "{TIPO} {NUM}/{ANO} —"
#
# 6. test_token_limits
#    - parent não deve ultrapassar PARENT_MAX_TOKENS
#    - child não deve ultrapassar CHILD_MAX_TOKENS
