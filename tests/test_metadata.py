"""
Módulo 1.9 — Testes de consolidação de metadados.
Rode com: pytest tests/test_metadata.py -v
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR


def _load_consolidate() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "consolidate",
        Path(__file__).parent.parent / "src" / "01_consolidate_metadata.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── helpers para criar dados de teste ─────────────────────────────────────────

def make_json(entries: list[dict]) -> dict:
    """Cria estrutura JSON no formato ANEEL a partir de uma lista de entradas planas."""
    registros = []
    for e in entries:
        registros.append({
            "numeracaoItem": "1.",
            "titulo": e.get("titulo", ""),
            "autor": e.get("autor", "ANEEL"),
            "material": "Legislação",
            "esfera": "Esfera:Federal",
            "situacao": f"Situação:{e.get('situacao', 'NÃO CONSTA REVOGAÇÃO EXPRESSA')}",
            "assinatura": f"Assinatura:{e.get('assinatura', '')}" if e.get("assinatura") else None,
            "publicacao": f"Publicação:{e.get('publicacao', '01/01/2022')}",
            "assunto": f"Assunto:{e.get('assunto', 'Geral')}",
            "ementa": e.get("ementa", "Ementa de teste"),
            "pdfs": [
                {
                    "tipo": "Texto Integral:",
                    "url": e.get("url", f"https://example.com/{e['arquivo']}"),
                    "arquivo": e["arquivo"],
                    "baixado": e.get("baixado", False),
                }
            ],
        })
    return {"2022-01-01": {"status": f"{len(registros)} registro(s).", "registros": registros}}


def write_json(tmp_path: Path, data: dict, name: str) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ── unit tests das funções auxiliares ─────────────────────────────────────────

def test_strip_label():
    m = _load_consolidate()
    assert m.strip_label("Situação:NÃO CONSTA REVOGAÇÃO EXPRESSA") == "NÃO CONSTA REVOGAÇÃO EXPRESSA"
    assert m.strip_label("Publicação:30/12/2022") == "30/12/2022"
    assert m.strip_label("Sem prefixo") == "Sem prefixo"
    assert m.strip_label(None) == ""
    assert m.strip_label("") == ""


def test_normalize_date_dmy():
    m = _load_consolidate()
    assert m.normalize_date("30/12/2022") == "2022-12-30"
    assert m.normalize_date("01/01/2016") == "2016-01-01"


def test_normalize_date_iso():
    m = _load_consolidate()
    assert m.normalize_date("2021-07-15") == "2021-07-15"


def test_normalize_date_invalid():
    m = _load_consolidate()
    assert m.normalize_date(None) is None
    assert m.normalize_date("") is None
    assert m.normalize_date("32/13/2022") is None
    assert m.normalize_date("texto") is None


def test_normalize_date_strips_prefix():
    m = _load_consolidate()
    assert m.normalize_date("Publicação:30/12/2022") == "2022-12-30"


def test_extract_tipo_ren():
    m = _load_consolidate()
    r = m.extract_tipo_info("ren20221054.pdf")
    assert r["tipo_sigla"] == "REN"
    assert r["tipo_nome"] == "Resolução Normativa"
    assert r["numero"] == 1054
    assert r["ano_arquivo"] == 2022
    assert r["numero_norm"] == "1054"


def test_extract_tipo_dsp_with_suffix():
    m = _load_consolidate()
    r = m.extract_tipo_info("dsp2022021spde.pdf")
    assert r["tipo_sigla"] == "DSP"
    assert r["numero"] == 21
    assert r["ano_arquivo"] == 2022


def test_extract_tipo_unknown():
    m = _load_consolidate()
    r = m.extract_tipo_info("xyzabc2022001.pdf")
    assert r["tipo_sigla"] == "OUTRO"


def test_is_revogada_true():
    m = _load_consolidate()
    assert m.is_revogada("REVOGADA pela REN 500/2020") is True
    assert m.is_revogada("Norma cancelada") is True


def test_is_revogada_false():
    m = _load_consolidate()
    assert m.is_revogada("NÃO CONSTA REVOGAÇÃO EXPRESSA") is False


# ── testes do parse_year_json ──────────────────────────────────────────────────

def test_parse_year_json_basic(tmp_path):
    m = _load_consolidate()
    data = make_json([
        {"arquivo": "ren20221054.pdf", "titulo": "REN 1054/2022",
         "publicacao": "01/12/2022", "assinatura": "28/11/2022"},
        {"arquivo": "dsp2022100.pdf", "titulo": "DSP 100/2022",
         "publicacao": "01/12/2022"},
    ])
    p = write_json(tmp_path, data, "test_2022.json")
    rows = m.parse_year_json(p, 2022)

    assert len(rows) == 2
    ren = next(r for r in rows if r["doc_id"] == "ren20221054")
    assert ren["tipo_sigla"] == "REN"
    assert ren["numero"] == 1054
    assert ren["data_pub"] == "2022-12-01"
    assert ren["data_ass"] == "2022-11-28"


def test_parse_year_json_skips_html(tmp_path):
    m = _load_consolidate()
    data = {"2022-01-01": {"status": "1 registro(s).", "registros": [
        {
            "numeracaoItem": "1.", "titulo": "REN 1054/2022", "autor": "ANEEL",
            "material": "Legislação", "esfera": "Esfera:Federal",
            "situacao": "Situação:NÃO CONSTA REVOGAÇÃO EXPRESSA",
            "assinatura": None, "publicacao": "Publicação:01/12/2022",
            "assunto": "Assunto:Distribuição", "ementa": "Ementa teste",
            "pdfs": [
                {"tipo": "Texto Integral:", "url": "http://x.com/ren20221054.html",
                 "arquivo": "ren20221054.html", "baixado": False},
                {"tipo": "Texto Integral:", "url": "http://x.com/ren20221054.pdf",
                 "arquivo": "ren20221054.pdf", "baixado": True},
            ],
        }
    ]}}
    p = write_json(tmp_path, data, "test_html.json")
    rows = m.parse_year_json(p, 2022)

    assert len(rows) == 1
    assert rows[0]["doc_id"] == "ren20221054"


def test_parse_year_json_ementa_cleans_imprimir(tmp_path):
    m = _load_consolidate()
    data = make_json([
        {"arquivo": "ren20221001.pdf", "ementa": "Estabelece regras. Imprimir"}
    ])
    p = write_json(tmp_path, data, "test_ementa.json")
    rows = m.parse_year_json(p, 2022)
    assert rows[0]["ementa"] == "Estabelece regras."


def test_parse_year_json_revogada_flag(tmp_path):
    m = _load_consolidate()
    data = make_json([
        {"arquivo": "ren20160001.pdf", "situacao": "REVOGADA pela REN 500/2021"},
        {"arquivo": "ren20160002.pdf", "situacao": "NÃO CONSTA REVOGAÇÃO EXPRESSA"},
    ])
    p = write_json(tmp_path, data, "test_revogada.json")
    rows = m.parse_year_json(p, 2016)

    rev = {r["doc_id"]: r["revogada"] for r in rows}
    assert rev["ren20160001"] is True
    assert rev["ren20160002"] is False


# ── testes do pipeline completo (requerem JSONs reais) ───────────────────────

def _load_real_parquet() -> pd.DataFrame:
    p = PROCESSED_DIR / "metadata.parquet"
    if not p.exists():
        pytest.skip("metadata.parquet não gerado — rode 01_consolidate_metadata.py primeiro")
    return pd.read_parquet(p)


def test_no_duplicate_doc_ids():
    df = _load_real_parquet()
    dupes = df["doc_id"].duplicated().sum()
    assert dupes == 0, f"{dupes} doc_ids duplicados encontrados"


def test_tipo_sigla_known_values():
    df = _load_real_parquet()
    known = {"REN", "REH", "REA", "DSP", "PRT", "OUTRO"}
    unknown = set(df["tipo_sigla"].unique()) - known
    assert not unknown, f"Tipos desconhecidos: {unknown}"


def test_data_pub_format():
    df = _load_real_parquet()
    iso_re = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    non_null = df["data_pub"].dropna()
    bad = non_null[~non_null.str.match(iso_re)]
    assert len(bad) == 0, f"Datas fora do formato ISO: {bad.head().tolist()}"


def test_doc_id_regex():
    # doc_ids devem começar com letras seguidas de ano de 4 dígitos
    # alguns documentos têm padrão 'sn' (sem número) após o ano — aceito mas deve ser OUTRO
    df = _load_real_parquet()
    base_pattern = re.compile(r'^[a-z]+\d{4}')
    bad = df[~df["doc_id"].str.match(base_pattern)]
    assert len(bad) == 0, f"doc_ids fora do padrão: {bad['doc_id'].head().tolist()}"
    # doc_ids com tipo canônico devem ter número após o ano
    canonical = df[df["tipo_sigla"] != "OUTRO"]
    num_pattern = re.compile(r'^[a-z]+\d{4}\d+')
    bad_canonical = canonical[~canonical["doc_id"].str.match(num_pattern)]
    assert len(bad_canonical) == 0, (
        f"doc_ids canônicos sem número: {bad_canonical['doc_id'].head().tolist()}"
    )
