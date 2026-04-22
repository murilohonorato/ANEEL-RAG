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
          data/processed/metadata.parquet  (fonte de verdade)
Saída:    data/processed/texts/{doc_id}.txt
          data/processed/parsed_docs.parquet

Uso:
    python src/03_parse.py
    python src/03_parse.py --tipo REN   # só resoluções normativas
    python src/03_parse.py --sample 50  # amostra para teste rápido
"""

import argparse
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

# ── garante que src/ está no path quando rodado standalone ────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PDFS_DIR, PROCESSED_DIR
from src.utils.logger import logger

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PDF_DIR           = PDFS_DIR
OUTPUT_DIR        = PROCESSED_DIR / "texts"
PARSED_DOCS_PATH  = PROCESSED_DIR / "parsed_docs.parquet"
METADATA_PATH     = PROCESSED_DIR / "metadata.parquet"
MIN_CHARS_PER_PAGE = 100   # abaixo disso → página escaneada

# Padrões de estrutura legal (compilados uma vez)
ART_RE  = re.compile(r'(?:^|\n)\s*Art\.?\s*\d+', re.MULTILINE)
PAR_RE  = re.compile(r'(?:^|\n)\s*§\s*\d+', re.MULTILINE)
INC_RE  = re.compile(r'(?:^|\n)\s*[IVX]+\s*[—–-]', re.MULTILINE)
ALIN_RE = re.compile(r'(?:^|\n)\s*[a-z]\)', re.MULTILINE)
ANEX_RE = re.compile(r'(?:^|\n)\s*ANEXO', re.MULTILINE)


# ── 3.1 — Detecção de tipo de PDF ─────────────────────────────────────────────

def detect_pdf_type(path: Path) -> Literal["native", "scanned", "mixed"]:
    """
    Detecta se o PDF tem texto nativo, é escaneado ou misto.

    Critério por página: contagem de caracteres via PyMuPDF.
    - native : >= 90% das páginas com >= MIN_CHARS_PER_PAGE chars
    - scanned: <= 10% das páginas com >= MIN_CHARS_PER_PAGE chars
    - mixed  : entre 10% e 90%
    """
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.warning(f"Não foi possível abrir {path.name}: {exc}")
        return "scanned"

    if doc.page_count == 0:
        doc.close()
        return "scanned"

    native_pages = 0
    scanned_pages = 0
    for page in doc:
        chars = len(page.get_text("text").strip())
        if chars >= MIN_CHARS_PER_PAGE:
            native_pages += 1
        else:
            scanned_pages += 1
    doc.close()

    total = native_pages + scanned_pages
    if total == 0:
        return "scanned"

    ratio = native_pages / total
    if ratio >= 0.9:
        return "native"
    if ratio <= 0.1:
        return "scanned"
    return "mixed"


# ── 3.2 — Extração de texto nativo (PyMuPDF) ──────────────────────────────────

def _fix_hyphenation(text: str) -> str:
    """Remove hifenização de fim de linha: 'pala-\\nvra' → 'palavra'."""
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)


def remove_control_chars(text: str) -> str:
    """Remove caracteres de controle exceto \\n e \\t."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def normalize_encoding(text: str) -> str:
    """Normaliza Unicode para NFC e remove BOM."""
    return unicodedata.normalize("NFC", text).replace('\ufeff', '')


def _clean_page_text(raw: str) -> str:
    """Aplica limpeza padronizada num texto de página."""
    text = _fix_hyphenation(raw)
    text = remove_control_chars(text)
    text = normalize_encoding(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_native(path: Path) -> list[dict]:
    """
    Extrai texto de PDF nativo com PyMuPDF.
    Retorna lista de dicts por página: {page, text, char_count}.
    """
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.error(f"PyMuPDF falhou em {path.name}: {exc}")
        return []

    pages: list[dict] = []
    for i, page in enumerate(doc):
        text = _clean_page_text(page.get_text("text"))
        pages.append({"page": i + 1, "text": text, "char_count": len(text)})
    doc.close()
    return pages


# ── 3.3 — OCR com Tesseract (PDFs escaneados) ─────────────────────────────────

def _configure_tesseract() -> None:
    """Configura o executável do Tesseract no Windows se necessário."""
    if sys.platform != "win32":
        return
    try:
        import pytesseract
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = candidate
                return
    except ImportError:
        pass


def extract_text_ocr(path: Path) -> list[dict]:
    """
    Extrai texto de PDF escaneado via pdf2image + Tesseract OCR.
    Retorna lista de dicts por página: {page, text, char_count}.
    Retorna [] se as dependências de OCR não estiverem instaladas.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as exc:
        logger.warning(f"Dependência OCR ausente ({exc}) — pulando OCR para {path.name}")
        return []

    _configure_tesseract()

    try:
        images = convert_from_path(str(path), dpi=200)
    except Exception as exc:
        logger.error(f"pdf2image falhou em {path.name}: {exc}")
        return []

    pages: list[dict] = []
    for i, img in enumerate(images):
        try:
            raw = pytesseract.image_to_string(img, lang="por+eng", config="--psm 6")
        except Exception as exc:
            logger.warning(f"OCR falhou na página {i + 1} de {path.name}: {exc}")
            raw = ""

        text = _clean_page_text(raw)
        pages.append({"page": i + 1, "text": text, "char_count": len(text)})

    return pages


# ── 3.4 — Extração de tabelas (pdfplumber) ────────────────────────────────────

def table_to_markdown(table: list[list]) -> str:
    """Converte tabela (lista de listas) para Markdown com pipe. Exposta para testes."""
    if not table:
        return ""

    # limpar células None e converter para str
    rows: list[list[str]] = []
    for row in table:
        if row:
            rows.append([str(c).strip() if c is not None else "" for c in row])

    if not rows:
        return ""

    header = rows[0]
    sep = ["-" * max(len(c), 3) for c in header]
    n_cols = len(header)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows[1:]:
        # garantir número correto de colunas
        row = (row + [""] * n_cols)[:n_cols]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_tables_as_markdown(path: Path) -> list[dict]:
    """
    Extrai tabelas de cada página usando pdfplumber.
    Retorna lista de {page, table_index, markdown}.
    Retorna [] se pdfplumber não estiver instalado ou falhar.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber não instalado — tabelas não serão extraídas.")
        return []

    results: list[dict] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables() or []
                    for j, table in enumerate(tables):
                        md = table_to_markdown(table)
                        if md:
                            results.append({"page": i + 1, "table_index": j, "markdown": md})
                except Exception as exc:
                    logger.debug(f"Tabela p.{i + 1} de {path.name}: {exc}")
    except Exception as exc:
        logger.warning(f"pdfplumber falhou em {path.name}: {exc}")

    return results


# ── 3.5 — Detecção de estrutura legal ─────────────────────────────────────────

def detect_legal_structure(text: str) -> dict:
    """
    Identifica elementos de estrutura legal no texto.
    Retorna dict com flags booleanas e contagens.
    """
    articles  = ART_RE.findall(text)
    paragraphs = PAR_RE.findall(text)
    incisos   = INC_RE.findall(text)
    alineas   = ALIN_RE.findall(text)
    annexes   = ANEX_RE.findall(text)
    return {
        "has_articles":    len(articles) > 0,
        "article_count":   len(articles),
        "has_paragraphs":  len(paragraphs) > 0,
        "paragraph_count": len(paragraphs),
        "has_incisos":     len(incisos) > 0,
        "inciso_count":    len(incisos),
        "has_alineas":     len(alineas) > 0,
        "has_annexes":     len(annexes) > 0,
    }


# ── 3.6 — Limpeza de texto final ──────────────────────────────────────────────

def remove_repeated_headers_footers(pages: list[dict]) -> list[dict]:
    """
    Remove linhas que aparecem em >= 50% das páginas (cabeçalhos/rodapés).
    Requer ao menos 3 páginas para calcular frequências com sentido.
    """
    if len(pages) < 3:
        return pages

    # contar frequência de cada linha única por documento
    line_counts: Counter = Counter()
    for p in pages:
        seen: set[str] = set()
        for line in p["text"].splitlines():
            stripped = line.strip()
            if len(stripped) > 5 and stripped not in seen:
                line_counts[stripped] += 1
                seen.add(stripped)

    threshold = len(pages) * 0.5
    repeated = {line for line, cnt in line_counts.items() if cnt >= threshold}

    if not repeated:
        return pages

    logger.debug(f"Removendo {len(repeated)} linhas repetitivas (cabeçalhos/rodapés)")
    cleaned: list[dict] = []
    for p in pages:
        new_lines = [l for l in p["text"].splitlines() if l.strip() not in repeated]
        new_text = "\n".join(new_lines).strip()
        cleaned.append({**p, "text": new_text, "char_count": len(new_text)})
    return cleaned


def normalize_punctuation(text: str) -> str:
    """Converte aspas curvas, travessões e NBSP para forma canônica."""
    replacements = {
        '\u201c': '"',  '\u201d': '"',   # aspas inglesas duplas
        '\u2018': "'",  '\u2019': "'",   # aspas inglesas simples
        '\u2013': ' – ',                 # en dash
        '\u2014': ' — ',                 # em dash
        '\u00a0': ' ',                   # non-breaking space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def assemble_full_text(pages: list[dict]) -> str:
    """Concatena páginas e aplica limpeza final."""
    full = "\n\n".join(p["text"] for p in pages if p["text"])
    full = normalize_punctuation(full)
    full = normalize_encoding(full)
    full = re.sub(r'\n{3,}', '\n\n', full)
    full = re.sub(r'[ \t]+', ' ', full)
    return full.strip()


# ── Pipeline por documento ────────────────────────────────────────────────────

def parse_document(
    doc_id: str,
    pdf_path: Path,
    tipo_sigla: str,
    numero: Optional[int],
    ano: int,
) -> dict:
    """
    Processa um único PDF e retorna dict com texto extraído e metadados de parse.
    O campo 'full_text' contém o texto completo limpo.
    """
    base: dict = {
        "doc_id":          doc_id,
        "tipo_sigla":      tipo_sigla,
        "numero":          numero,
        "ano":             ano,
        "full_text":       "",
        "page_count":      0,
        "char_count":      0,
        "pdf_type":        "unknown",
        "has_tables":      False,
        "has_articles":    False,
        "article_count":   0,
        "parse_method":    "failed",
        "parse_timestamp": datetime.now().isoformat(),
        "error":           None,
    }

    if not pdf_path.exists():
        base["error"] = "arquivo_nao_encontrado"
        logger.warning(f"{doc_id}: PDF não encontrado em {pdf_path}")
        return base

    # 3.1 — detectar tipo
    pdf_type = detect_pdf_type(pdf_path)
    base["pdf_type"] = pdf_type

    pages: list[dict] = []

    # 3.2 — extração nativa
    if pdf_type in ("native", "mixed"):
        pages = extract_text_native(pdf_path)
        base["parse_method"] = "native"

    # 3.3 — OCR para escaneados ou páginas sem texto em PDF misto
    if pdf_type == "scanned":
        ocr_pages = extract_text_ocr(pdf_path)
        if ocr_pages:
            pages = ocr_pages
            base["parse_method"] = "ocr"

    elif pdf_type == "mixed" and pages:
        # substituir páginas nativas pobres por OCR
        ocr_pages = extract_text_ocr(pdf_path)
        if ocr_pages:
            for i, p in enumerate(pages):
                if p["char_count"] < MIN_CHARS_PER_PAGE and i < len(ocr_pages):
                    pages[i] = ocr_pages[i]
            base["parse_method"] = "mixed"

    if not pages:
        base["error"] = "sem_texto"
        return base

    # 3.6 — remover cabeçalhos/rodapés repetitivos
    pages = remove_repeated_headers_footers(pages)

    # 3.4 — tabelas (somente PDFs com texto nativo)
    if pdf_type in ("native", "mixed"):
        tables = extract_tables_as_markdown(pdf_path)
        base["has_tables"] = len(tables) > 0

    # montar texto completo
    full_text = assemble_full_text(pages)
    if not full_text:
        base["error"] = "texto_vazio_pos_limpeza"
        return base

    # 3.5 — detectar estrutura legal
    structure = detect_legal_structure(full_text)

    base.update({
        "full_text":    full_text,
        "page_count":   len(pages),
        "char_count":   len(full_text),
        "has_articles": structure["has_articles"],
        "article_count": structure["article_count"],
    })
    return base


# ── Pipeline batch ────────────────────────────────────────────────────────────

def run_parse(
    tipo_filter: Optional[str] = None,
    sample: Optional[int] = None,
) -> pd.DataFrame:
    """
    Processa todos os PDFs do metadata.parquet e salva parsed_docs.parquet.
    Suporta modo incremental: pula doc_ids já presentes no parquet existente.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.exists():
        logger.error(
            f"metadata.parquet não encontrado em {METADATA_PATH}. "
            "Rode src/01_consolidate_metadata.py primeiro."
        )
        return pd.DataFrame()

    df_meta = pd.read_parquet(METADATA_PATH)
    logger.info(f"Metadados carregados: {len(df_meta)} PDFs")

    if tipo_filter:
        df_meta = df_meta[df_meta["tipo_sigla"] == tipo_filter.upper()]
        logger.info(f"Filtro tipo={tipo_filter.upper()}: {len(df_meta)} PDFs")

    if sample:
        df_meta = df_meta.sample(min(sample, len(df_meta)), random_state=42)
        logger.info(f"Amostra: {len(df_meta)} PDFs")

    # modo incremental — pular apenas os que foram processados com sucesso
    already_done: set[str] = set()
    if PARSED_DOCS_PATH.exists():
        existing = pd.read_parquet(PARSED_DOCS_PATH, columns=["doc_id", "error"])
        # considera "feito" apenas quem não tem erro
        done_ok = existing[existing["error"].isna()]["doc_id"].tolist()
        already_done = set(done_ok)
        failed_before = existing[existing["error"].notna()]["doc_id"].tolist()
        logger.info(f"Ja processados com sucesso (incremental): {len(already_done)}")
        if failed_before:
            logger.info(f"Reprocessando {len(failed_before)} que falharam anteriormente")
        df_meta = df_meta[~df_meta["doc_id"].isin(already_done)]
        logger.info(f"Restam para processar: {len(df_meta)}")

    if df_meta.empty:
        logger.info("Nada a processar.")
        return pd.DataFrame()

    results: list[dict] = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Parsing PDFs"):
        doc_id   = row["doc_id"]
        ano      = int(row["ano"])
        pdf_path = PDF_DIR / str(ano) / f"{doc_id}.pdf"

        parsed = parse_document(
            doc_id=doc_id,
            pdf_path=pdf_path,
            tipo_sigla=str(row["tipo_sigla"]),
            numero=row.get("numero"),
            ano=ano,
        )

        # salvar .txt individual
        if parsed["full_text"]:
            txt_path = OUTPUT_DIR / f"{doc_id}.txt"
            txt_path.write_text(parsed["full_text"], encoding="utf-8")
            parsed["txt_path"] = str(txt_path)
        else:
            parsed["txt_path"] = None

        # não persistir full_text no parquet (economizar espaço; usar .txt)
        results.append({k: v for k, v in parsed.items() if k != "full_text"})

    if not results:
        logger.warning("Nenhum resultado gerado.")
        return pd.DataFrame()

    df_new = pd.DataFrame(results)

    # concatenar com existentes — remover registros antigos que foram reprocessados
    if PARSED_DOCS_PATH.exists():
        df_old = pd.read_parquet(PARSED_DOCS_PATH)
        # remover do old os doc_ids que foram reprocessados agora
        reprocessed_ids = set(df_new["doc_id"].tolist())
        df_old = df_old[~df_old["doc_id"].isin(reprocessed_ids)]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_parquet(PARSED_DOCS_PATH, index=False)
    logger.info(f"parsed_docs.parquet salvo: {len(df_final)} documentos")
    _log_stats(df_new)

    return df_final


def _log_stats(df: pd.DataFrame) -> None:
    if df.empty:
        return
    total   = len(df)
    failed  = int(df["error"].notna().sum())
    by_type = df["pdf_type"].value_counts().to_dict()
    with_art = int(df["has_articles"].sum())
    with_tab = int(df["has_tables"].sum())
    logger.info(
        f"Parse stats — total={total}, falhas={failed}, "
        f"tipos={by_type}, com_artigos={with_art}, com_tabelas={with_tab}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Módulo 3 — Parsing de PDFs ANEEL")
    parser.add_argument("--tipo",   type=str, help="Filtrar por tipo (ex: REN, DSP)")
    parser.add_argument("--sample", type=int, help="Processar apenas N documentos (teste)")
    args = parser.parse_args()

    df = run_parse(tipo_filter=args.tipo, sample=args.sample)
    if not df.empty:
        cols = ["doc_id", "tipo_sigla", "ano", "pdf_type", "parse_method",
                "has_articles", "article_count", "has_tables", "char_count"]
        print(df[cols].head(10).to_string())
