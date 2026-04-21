"""
01_consolidate_metadata.py
--------------------------
Lê os 3 JSONs da ANEEL e gera data/processed/metadata.parquet.

Cada linha = 1 PDF com seus metadados normalizados.
Este parquet é a fonte de verdade para os passos seguintes.

Uso:
    python src/01_consolidate_metadata.py
"""

import json
import re
import sys
from pathlib import Path
from datetime import date
from typing import Optional

import pandas as pd

# ── garante que src/ está no path quando rodado standalone ────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import JSON_FILES, PROCESSED_DIR, PDFS_DIR
from src.utils.logger import logger

# CONFIG
OUTPUT_PARQUET = PROCESSED_DIR / "metadata.parquet"
OUTPUT_CSV = PROCESSED_DIR / "metadata.csv"
OUTPUT_STATS = PROCESSED_DIR / "index_stats.json"

TIPO_MAP: dict[str, tuple[str, str]] = {
    "ren":  ("REN", "Resolução Normativa"),
    "reh":  ("REH", "Resolução Homologatória"),
    "rea":  ("REA", "Resolução Autorizativa"),
    "dsp":  ("DSP", "Despacho"),
    "prt":  ("PRT", "Portaria"),
}

REVOGACAO_TERMS = ("revogad", "cancelad", "substituíd", "substituid")

DATE_PREFIXES = ("publicação:", "publicacao:", "assinatura:", "data:")


# ── helpers ───────────────────────────────────────────────────────────────────

def strip_label(text: Optional[str]) -> str:
    """Remove prefixo 'Campo:' de campos do JSON. Ex: 'Situação:Foo' → 'Foo'."""
    if not text:
        return ""
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text.strip()


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Converte 'DD/MM/YYYY' ou 'YYYY-MM-DD' → 'YYYY-MM-DD'. Retorna None se inválido."""
    if not raw:
        return None
    raw = strip_label(raw).strip()
    # DD/MM/YYYY
    m = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', raw)
    if m:
        try:
            d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            return d.isoformat()
        except ValueError:
            return None
    # YYYY-MM-DD
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', raw)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d.isoformat()
        except ValueError:
            return None
    return None


def extract_tipo_info(arquivo: str) -> dict:
    """
    Extrai tipo_sigla, tipo_nome, numero e ano do nome do arquivo PDF.
    Ex: 'ren20221054.pdf' → {'tipo_sigla': 'REN', 'tipo_nome': 'Resolução Normativa',
                              'numero': 1054, 'numero_norm': '1054', 'ano_arquivo': 2022}
    """
    name = Path(arquivo).stem.lower()
    m = re.match(r'^([a-z]+)(\d{4})(\d+)', name)
    if not m:
        return {
            "tipo_sigla": "OUTRO", "tipo_nome": "Outro",
            "numero": None, "numero_norm": None, "ano_arquivo": None,
        }
    prefix = m.group(1)
    ano_arquivo = int(m.group(2))
    numero = int(m.group(3))
    numero_norm = f"{numero:04d}"

    sigla, nome = TIPO_MAP.get(prefix, ("OUTRO", "Outro"))
    return {
        "tipo_sigla": sigla,
        "tipo_nome": nome,
        "numero": numero,
        "numero_norm": numero_norm,
        "ano_arquivo": ano_arquivo,
    }


def is_revogada(situacao: str) -> bool:
    s = situacao.lower()
    return any(t in s for t in REVOGACAO_TERMS)


def parse_year_json(path: Path, year: int) -> list[dict]:
    """
    Lê um JSON de ano e retorna lista plana de dicts, um por PDF.
    Cada dict corresponde a um arquivo PDF com todos os metadados do registro pai.
    """
    logger.info(f"Lendo {path.name} ...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict] = []
    anomalies: list[str] = []

    for date_key, day_data in data.items():
        registros = day_data.get("registros", [])
        for reg in registros:
            # campos do registro
            titulo = (reg.get("titulo") or "").strip()
            autor = (reg.get("autor") or "").strip()
            esfera = strip_label(reg.get("esfera") or "")
            situacao_raw = strip_label(reg.get("situacao") or "")
            assunto = strip_label(reg.get("assunto") or "")
            ementa_raw = (reg.get("ementa") or "").strip()
            # remover "Imprimir" no fim da ementa
            ementa = re.sub(r'\s*Imprimir\s*$', '', ementa_raw).strip()

            data_pub_raw = strip_label(reg.get("publicacao") or "")
            data_ass_raw = strip_label(reg.get("assinatura") or "")
            data_pub = normalize_date(data_pub_raw) or normalize_date(date_key)
            data_ass = normalize_date(data_ass_raw)

            revogada = is_revogada(situacao_raw)

            pdfs = reg.get("pdfs") or []
            if not pdfs:
                logger.warning(f"[{year}/{date_key}] Registro sem PDFs: '{titulo}'")
                continue

            for pdf in pdfs:
                arquivo = (pdf.get("arquivo") or "").strip()
                if not arquivo.lower().endswith(".pdf"):
                    continue  # pular HTML e outros formatos

                url = (pdf.get("url") or "").strip()
                baixado_json = bool(pdf.get("baixado", False))
                pdf_tipo_doc = strip_label(pdf.get("tipo") or "")

                tipo_info = extract_tipo_info(arquivo)
                tipo_sigla = tipo_info["tipo_sigla"]
                tipo_nome = tipo_info["tipo_nome"]
                numero = tipo_info["numero"]
                numero_norm = tipo_info["numero_norm"]
                ano_arquivo = tipo_info["ano_arquivo"] or year

                # doc_id canônico = nome do arquivo sem extensão (lower)
                doc_id = Path(arquivo).stem.lower()

                # normalizar numero_norm para o doc_id canônico do roadmap
                # (tipo_sigla.lower() + ano + numero_norm), mas doc_id real vem do arquivo
                pdf_filename = f"{doc_id}.pdf"
                pdf_path_expected = str(PDFS_DIR / str(ano_arquivo) / pdf_filename)
                pdf_baixado = Path(pdf_path_expected).exists() or baixado_json

                rows.append({
                    "doc_id": doc_id,
                    "tipo_sigla": tipo_sigla,
                    "tipo_nome": tipo_nome,
                    "numero": numero,
                    "numero_norm": numero_norm,
                    "ano": ano_arquivo,
                    "ano_json": year,
                    "titulo": titulo,
                    "autor": autor,
                    "esfera": esfera,
                    "situacao": situacao_raw,
                    "revogada": revogada,
                    "assunto": assunto,
                    "ementa": ementa,
                    "data_pub": data_pub,
                    "data_ass": data_ass,
                    "pdf_url": url,
                    "pdf_arquivo": arquivo,
                    "pdf_tipo_doc": pdf_tipo_doc,
                    "pdf_baixado": pdf_baixado,
                    "pdf_filename": pdf_filename,
                    "pdf_path_expected": pdf_path_expected,
                })

    logger.info(f"  → {len(rows)} PDFs extraídos de {year}")
    return rows


# ── pipeline principal ────────────────────────────────────────────────────────

def consolidate() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for year, path in sorted(JSON_FILES.items()):
        if not path.exists():
            logger.warning(f"JSON não encontrado: {path} — pulando {year}")
            continue
        rows = parse_year_json(path, year)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("Nenhum dado carregado. Verifique os JSONs em data/raw/.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    before = len(df)

    # deduplicar por arquivo (lower) — anos mais recentes ficam, graças à ordem de iteração
    df["_arquivo_key"] = df["pdf_arquivo"].str.lower()
    df = df.drop_duplicates(subset=["_arquivo_key"], keep="last")
    df = df.drop(columns=["_arquivo_key"])
    dupes = before - len(df)
    if dupes:
        logger.info(f"Duplicatas removidas: {dupes}")

    # ordenar
    df = df.sort_values(["ano", "tipo_sigla", "numero"]).reset_index(drop=True)

    logger.info(f"Total final: {len(df)} PDFs únicos")

    # salvar
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Salvo: {OUTPUT_PARQUET}")

    # stats
    stats = {
        "total_registros": len(df),
        "por_tipo": df["tipo_sigla"].value_counts().to_dict(),
        "por_ano": df["ano"].value_counts().sort_index().to_dict(),
        "pdfs_baixados": int(df["pdf_baixado"].sum()),
        "pdfs_pendentes": int((~df["pdf_baixado"]).sum()),
        "revogadas": int(df["revogada"].sum()),
    }
    import json as _json
    OUTPUT_STATS.write_text(_json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(f"Stats: {stats}")

    # anomalias
    sem_ementa = df[df["ementa"].str.len() < 10]
    if len(sem_ementa):
        logger.warning(f"{len(sem_ementa)} documentos com ementa muito curta (<10 chars)")
    tipo_outro = df[df["tipo_sigla"] == "OUTRO"]
    if len(tipo_outro):
        logger.warning(f"{len(tipo_outro)} documentos com tipo_sigla=OUTRO")

    return df


if __name__ == "__main__":
    df = consolidate()
    if not df.empty:
        print(df[["doc_id", "tipo_sigla", "ano", "numero", "data_pub", "revogada"]].head(10).to_string())
