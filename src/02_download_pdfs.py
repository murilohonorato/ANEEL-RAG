"""
02_download_pdfs.py
-------------------
Download dos PDFs da ANEEL via Playwright.
Navega diretamente para cada URL — idêntico ao que o browser faz manualmente.

Instalação:
    pip install playwright tqdm
    playwright install chromium

Uso:
    cd "C:\\Users\\muril\\Downloads\\dados_grupo_estudos"
    python 02_download_pdfs.py
"""

import csv
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
JSON_FILES = [
    "biblioteca_aneel_gov_br_legislacao_2016_metadados.json",
    "biblioteca_aneel_gov_br_legislacao_2021_metadados.json",
    "biblioteca_aneel_gov_br_legislacao_2022_metadados.json",
]
OUTPUT_DIR   = Path("pdfs")
ERROR_LOG    = Path("erros_download.csv")
DELAY_S      = 2.0
TIMEOUT_MS   = 45_000
MAX_RETRIES  = 3
MIN_PDF_SIZE = 500
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("download.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


@dataclass
class PDFRecord:
    url: str
    arquivo: str
    tipo_doc: str
    ano: str

    @property
    def dest_path(self) -> Path:
        is_html = "planalto.gov.br" in self.url or self.url.endswith((".htm", ".html"))
        ext  = ".htm" if is_html else ".pdf"
        return OUTPUT_DIR / self.ano / (Path(self.arquivo).stem + ext)


def fix_url(url: str) -> str | None:
    url = url.strip()
    if not url:
        return None
    url = re.sub(r"^https?://\s+https?://", "http://", url)
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return url


def load_records() -> list[PDFRecord]:
    records, seen = [], set()
    for json_file in JSON_FILES:
        if not Path(json_file).exists():
            log.warning(f"Não encontrado: {json_file}")
            continue
        ano = json_file.split("_")[5]
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        for day_data in data.values():
            for reg in day_data.get("registros", []):
                titulo = reg.get("titulo", "")
                tipo = titulo.split(" - ")[0].strip() if " - " in titulo else "?"
                for pdf in reg.get("pdfs", []):
                    arquivo = pdf.get("arquivo", "").strip()
                    if not arquivo or arquivo in seen:
                        continue
                    seen.add(arquivo)
                    url = fix_url(pdf.get("url", ""))
                    if url:
                        records.append(PDFRecord(url=url, arquivo=arquivo, tipo_doc=tipo, ano=ano))
    log.info(f"Registros únicos: {len(records)}")
    return records


def filter_pending(records: list[PDFRecord]) -> list[PDFRecord]:
    pending, already = [], 0
    for r in records:
        if r.dest_path.exists() and r.dest_path.stat().st_size > MIN_PDF_SIZE:
            already += 1
        else:
            pending.append(r)
    if already:
        log.info(f"Já baixados (pulando): {already}")
    return pending


def download_all(records: list[PDFRecord]):
    OUTPUT_DIR.mkdir(exist_ok=True)
    error_rows = []
    ok = fail = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # headless=False passa melhor no Cloudflare
        context = browser.new_context(
            accept_downloads=True,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        # Aquece a sessão — visita o site e pega cookies do Cloudflare
        log.info("Iniciando sessão...")
        try:
            page.goto("http://www2.aneel.gov.br/cedoc/", timeout=TIMEOUT_MS, wait_until="domcontentloaded")
            time.sleep(3)
        except Exception as e:
            log.warning(f"Página inicial: {e}")

        for record in tqdm(records, desc="Baixando", unit="pdf", colour="green"):
            dest = record.dest_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            for attempt in range(MAX_RETRIES):
                try:
                    # Intercepta a resposta da navegação direta — igual ao browser abrindo o PDF
                    response = page.goto(record.url, timeout=TIMEOUT_MS, wait_until="commit")

                    if response is None:
                        raise Exception("Sem resposta")

                    if response.status == 404:
                        error_rows.append({"arquivo": record.arquivo, "url": record.url,
                                           "status": 404, "erro": "Not Found"})
                        break

                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")

                    raw = response.body()

                    if len(raw) < MIN_PDF_SIZE:
                        raise Exception(f"Arquivo pequeno: {len(raw)} bytes")

                    dest.write_bytes(raw)
                    ok += 1
                    break

                except Exception as exc:
                    wait = [5, 15, 30][min(attempt, 2)]
                    if attempt < MAX_RETRIES - 1:
                        log.debug(f"Retry {attempt+1} ({wait}s): {record.arquivo} — {exc}")
                        time.sleep(wait)
                    else:
                        log.warning(f"FALHOU {record.arquivo}: {exc}")
                        error_rows.append({"arquivo": record.arquivo, "url": record.url,
                                           "status": -1, "erro": str(exc)[:200]})
                        fail += 1

            time.sleep(DELAY_S)

        page.close()
        browser.close()

    if error_rows:
        with open(ERROR_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["arquivo", "url", "status", "erro"])
            writer.writeheader()
            writer.writerows(error_rows)
        log.info(f"Erros salvos em: {ERROR_LOG}")

    return ok, fail


def main():
    start = time.perf_counter()
    records = load_records()
    records = filter_pending(records)

    if not records:
        log.info("Nada para baixar.")
        return

    log.info(f"Para baixar: {len(records)} arquivos")
    log.info(f"Tempo estimado: ~{len(records) * DELAY_S / 3600:.1f}h")

    ok, fail = download_all(records)

    elapsed = time.perf_counter() - start
    log.info(
        f"\n{'─'*40}"
        f"\n✓ Baixados:  {ok}"
        f"\n✗ Falharam:  {fail}"
        f"\nTempo total: {elapsed/60:.1f} min"
        f"\n{'─'*40}"
    )


if __name__ == "__main__":
    main()
