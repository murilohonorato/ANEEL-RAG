"""
ANEEL PDF bulk downloader — Selenium + resumable checkpoint.
Configurado para: 2021

Fluxo:
  1. Na primeira execução: extrai todos os URLs do JSON → salva index.json
  2. Em toda execução: carrega index.json + progress.json, pula os já baixados
  3. Downloads via undetected-chromedriver (Chrome real, invisível a bot detection)

Arquivos gerados em extraction/:
  index_2021.json     — todos os PDFs com metadata + filename canônico
  progress_2021.json  — status por ID: "done" | "error:<msg>"
  pdfs_2021/          — pdf_1.pdf, pdf_2.pdf, ...

Install:
  pip install -r requirements.txt

Uso:
  cd extract_pdfs
  python extraction/download_pdfs_2021.py
"""

import json
import time
import random
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import undetected_chromedriver as uc

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
JSON_PATH     = Path(r"C:\Users\muril\Downloads\dados_grupo_estudos\biblioteca_aneel_gov_br_legislacao_2021_metadados.json")
OUTPUT_DIR    = BASE_DIR / "pdfs_2021"
INDEX_FILE    = BASE_DIR / "index_2021.json"
PROGRESS_FILE = BASE_DIR / "progress_2021.json"

DELAY_MIN        = 3.0   # segundos entre downloads
DELAY_MAX        = 7.0
DOWNLOAD_TIMEOUT = 40    # segundos esperando o arquivo aparecer no disco
SAVE_EVERY       = 10    # salva progress.json a cada N downloads
# ─────────────────────────────────────────────────────────────────────────────


def build_index(json_path: Path) -> list:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for date_key, day_data in data.items():
        if not isinstance(day_data, dict):
            continue
        for registro in day_data.get("registros", []):
            for pdf in registro.get("pdfs", []):
                url = pdf.get("url", "")
                if not url.lower().strip().endswith(".pdf"):
                    continue
                url = url.strip()
                idx = len(entries) + 1
                entries.append({
                    "id":       idx,
                    "filename": f"pdf_{idx}.pdf",
                    "url":      url,
                    "arquivo":  pdf.get("arquivo", "").strip(),
                    "titulo":   registro.get("titulo", ""),
                    "data":     date_key,
                    "tipo":     pdf.get("tipo", ""),
                })
    return entries


def load_index() -> list:
    if INDEX_FILE.exists():
        with open(INDEX_FILE, encoding="utf-8") as f:
            return json.load(f)

    print("[+] Construindo index_2021.json (primeira execução)...")
    entries = build_index(JSON_PATH)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"    {len(entries)} PDFs indexados → {INDEX_FILE}")
    return entries


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def make_driver() -> uc.Chrome:
    prefs = {
        "download.default_directory":    str(OUTPUT_DIR.resolve()),
        "download.prompt_for_download":  False,
        "download.directory_upgrade":    True,
        "plugins.always_open_pdf_externally": True,
    }
    options = uc.ChromeOptions()
    options.add_experimental_option("prefs", prefs)
    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(40)
    return driver


def wait_for_download(before: set, timeout: int = DOWNLOAD_TIMEOUT) -> Optional[Path]:
    """Aguarda um novo .pdf aparecer em OUTPUT_DIR."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        current     = set(OUTPUT_DIR.glob("*.pdf"))
        in_progress = set(OUTPUT_DIR.glob("*.crdownload"))
        new_files   = current - before
        if new_files and not in_progress:
            return new_files.pop()
        time.sleep(0.5)
    return None


def download_one(driver: uc.Chrome, entry: dict, progress: dict) -> bool:
    pdf_id = str(entry["id"])
    dest   = OUTPUT_DIR / entry["filename"]
    before = set(OUTPUT_DIR.glob("*.pdf"))

    try:
        driver.get(entry["url"])
    except Exception:
        # Timeout normal para PDFs — Chrome baixa sem "carregar" a página
        pass

    downloaded = wait_for_download(before)

    if downloaded:
        shutil.move(str(downloaded), str(dest))
        progress[pdf_id] = "done"
        return True
    else:
        progress[pdf_id] = f"error:timeout — {entry['url']}"
        return False


def print_stats(progress: dict, total: int) -> None:
    done   = sum(1 for v in progress.values() if v == "done")
    errors = sum(1 for v in progress.values() if v.startswith("error"))
    pct    = done / total * 100 if total else 0
    print(f"\n[stats] {done}/{total} baixados ({pct:.1f}%) | {errors} erros")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index    = load_index()
    progress = load_progress()

    total      = len(index)
    done_count = sum(1 for v in progress.values() if v == "done")
    pending    = [e for e in index if progress.get(str(e["id"]), "") != "done"]

    print(f"[+] Ano: 2021")
    print(f"[+] Total: {total} | Já baixados: {done_count} | Pendentes: {len(pending)}")
    print(f"[+] Salvando em: {OUTPUT_DIR.resolve()}")

    if not pending:
        print("[+] Tudo já baixado!")
        return

    driver = make_driver()
    start  = datetime.now()

    try:
        for i, entry in enumerate(pending, start=1):
            pdf_id = str(entry["id"])
            dest   = OUTPUT_DIR / entry["filename"]

            # Arquivo já no disco de run anterior
            if dest.exists():
                progress[pdf_id] = "done"
                print(f"[{i}/{len(pending)}] #{entry['id']} já existe, pulando")
                continue

            success = download_one(driver, entry, progress)
            status  = "✓" if success else "✗"

            elapsed = max((datetime.now() - start).seconds, 1)
            eta_s   = (elapsed / i) * (len(pending) - i)
            eta_h   = eta_s / 3600

            print(
                f"[{i}/{len(pending)}] {status} #{entry['id']}  {entry['arquivo'] or entry['filename']}"
                f"  ETA: {eta_h:.1f}h",
                flush=True,
            )

            if not success:
                print(f"  ! {progress[pdf_id]}", flush=True)

            if i % SAVE_EVERY == 0:
                save_progress(progress)

            if i < len(pending):
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    finally:
        save_progress(progress)
        driver.quit()

    print_stats(progress, total)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrompido. Progress salvo — rode novamente para continuar.")
        sys.exit(0)
