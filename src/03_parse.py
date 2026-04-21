"""
03_parse.py
-----------
Extrai texto limpo dos PDFs baixados.

Estratégia:
- PyMuPDF para PDFs com texto nativo
- Tesseract OCR para PDFs escaneados (detectados por baixa densidade de texto)
- Remove cabeçalhos/rodapés repetitivos
- Normaliza hifenização, espaços, encoding

Entrada:  data/pdfs/{ano}/{arquivo}.pdf
Saída:    data/processed/texts/{doc_id}.txt
          data/processed/parse_stats.parquet

Uso:
    python src/03_parse.py
    python src/03_parse.py --tipo REN  # só resoluções normativas
    python src/03_parse.py --sample 50  # amostra para teste
"""

# CONFIG
PDF_DIR      = "data/pdfs"
OUTPUT_DIR   = "data/processed/texts"
STATS_PATH   = "data/processed/parse_stats.parquet"
MIN_CHARS_PER_PAGE = 100   # abaixo disso → PDF escaneado → OCR
WORKERS      = 4           # processos paralelos

# TODO: implementar com Claude Code
