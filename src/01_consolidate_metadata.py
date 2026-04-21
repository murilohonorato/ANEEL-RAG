"""
01_consolidate_metadata.py
--------------------------
Lê os 3 JSONs da ANEEL e gera data/processed/metadata.parquet.

Cada linha = 1 PDF com seus metadados normalizados.
Este parquet é a fonte de verdade para os passos seguintes.

Uso:
    python src/01_consolidate_metadata.py
"""

# CONFIG
JSON_FILES = [
    "data/raw/biblioteca_aneel_gov_br_legislacao_2016_metadados.json",
    "data/raw/biblioteca_aneel_gov_br_legislacao_2021_metadados.json",
    "data/raw/biblioteca_aneel_gov_br_legislacao_2022_metadados.json",
]
OUTPUT_PATH = "data/processed/metadata.parquet"

# TODO: implementar com Claude Code
# Campos esperados no parquet:
# doc_id, ano, data_publicacao, data_assinatura, tipo_sigla, tipo_nome,
# numero, titulo, autor, esfera, situacao, assunto, ementa,
# pdf_url, pdf_arquivo, pdf_tipo_doc, baixado
