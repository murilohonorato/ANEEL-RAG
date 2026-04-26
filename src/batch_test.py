"""
batch_test.py
-------------
Executa um lote de perguntas no pipeline RAG e gera um relatório Markdown
com respostas, fontes e timing de cada query.

Uso:
    python src/batch_test.py
    python src/batch_test.py --no-rerank   # pula reranking
    python src/batch_test.py --debug        # scores e detalhes de retrieval
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# garante UTF-8 no terminal Windows (necessário para caracteres como №)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from qdrant_client import QdrantClient

from src.config import QDRANT_PATH, RESULTS_DIR
from src.utils.logger import logger

# ── Perguntas de teste ────────────────────────────────────────────────────────

QUESTIONS = [
    "Qual é a capacidade instalada da unidade geradora UG1 liberada para operação em teste na EOL Vila Espírito Santo IV e a partir de qual data essa operação foi autorizada?",
    "Qual é o município, o estado e a razão social da empresa titular da EOL Vila Espírito Santo IV (antiga Potiguar B24), conforme o Despacho Nº 4.194/2021?",
    "De acordo com a tabela de 'Características do Projeto' da Portaria N° 264/2016, qual é a capacidade instalada total e o número de unidades geradoras que compõem a EOL Diamante III?",
    "Qual é o CNPJ e a Razão Social da 'Pessoa Jurídica Controladora' listada na tabela do projeto da EOL Diamante III (Portaria N° 264/2016)?",
    "Segundo a Tabela 1 do Despacho № 3.399/2016, quais são os nomes exatos de todas as pessoas físicas e da pessoa jurídica que propuseram o Projeto Básico da PCH COR 125?",
    "O Despacho № 3.399/2016 (PCH COR 125) condiciona a homologação dos parâmetros de Garantia Física à apresentação de quais documentos específicos à ANEEL?",
    "Com base na tabela de inadimplência do Edital de Notificação de 26/12/2016, qual é o valor exato (em R$) cobrado da 'Ribeirão Energia Ltda.' e qual é o número do Ofício (SAF/ANEEL) emitido para ela?",
    "De acordo com as instruções textuais do Edital de Notificação da TFSEE (2016), quais são os dois telefones e o e-mail disponibilizados para contato caso o pagamento do encargo já tenha sido processado?",
    "Conforme a tabela de características técnicas contida no Despacho № 3.407/2016, quais são as coordenadas geográficas exatas do eixo do barramento e qual é a área do reservatório da PCH Eixo 1?",
    "O Despacho № 3.407/2016 (PCH Eixo 1) estabelece uma regra sobre o registro de intenção à outorga (DRI-PCH) antes da entrega do Sumário Executivo. O que o texto define sobre a titularidade do DRI-PCH e sobre novas solicitações para o mesmo aproveitamento?",
]


# ── Formatação do relatório Markdown ─────────────────────────────────────────

def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled) + f" {score:.3f}"


def build_markdown_report(results: list[dict], args_info: dict) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# ANEEL RAG — Relatório de Teste em Lote",
        f"",
        f"**Data:** {now_str}  ",
        f"**Reranking:** {'sim' if args_info.get('rerank') else 'não'}  ",
        f"**Total de queries:** {len(results)}",
        "",
        "---",
        "",
    ]

    for i, res in enumerate(results, 1):
        q        = res["question"]
        answer   = res.get("answer", "*(sem resposta)*")
        filters  = res.get("filters", {})
        contexts = res.get("contexts", [])
        timing   = res.get("timing", {})
        error    = res.get("error")

        lines += [
            f"## Query {i}",
            "",
            f"**Pergunta:** {q}",
            "",
        ]

        # filtros extraídos
        active = {k: v for k, v in filters.items() if v is not None}
        if active:
            filter_str = " | ".join(f"`{k}={v}`" for k, v in active.items())
            lines += [f"**Filtros extraídos:** {filter_str}", ""]

        critic  = res.get("critic")

        if error:
            lines += [f"> ⚠️ **Erro:** {error}", ""]
        else:
            critic_tag = ""
            if critic:
                if critic.get("valid"):
                    critic_tag = f" ✅ crítico: `{critic.get('reason', 'ok')}`"
                else:
                    critic_tag = f" ⚠️ crítico rejeitou: `{critic.get('reason', '')}`"
            lines += [
                f"**Resposta:**{critic_tag}",
                "",
                answer,
                "",
            ]

            # fontes
            if contexts:
                lines += [
                    "**Fontes utilizadas:**",
                    "",
                    "| # | Documento | Data Pub | Situação | Score Rerank | Score RRF |",
                    "|---|-----------|----------|----------|-------------|-----------|",
                ]
                for j, ctx in enumerate(contexts, 1):
                    tipo    = ctx.get("tipo_sigla", "?")
                    numero  = ctx.get("numero")
                    ano     = ctx.get("ano", "?")
                    num_str = str(int(numero)) if numero and str(numero) != "nan" else "s/n"
                    doc     = f"{tipo} {num_str}/{ano}"
                    data    = ctx.get("data_pub", "?")
                    sit     = ctx.get("situacao", "?")
                    if sit and len(sit) > 35:
                        sit = sit[:35] + "…"
                    rr  = ctx.get("rerank_score")
                    rrf = ctx.get("rrf_score", 0)
                    rr_str  = f"{rr:.3f}" if rr is not None else "—"
                    rrf_str = f"{rrf:.4f}"
                    lines.append(f"| {j} | **{doc}** | {data} | {sit} | {rr_str} | {rrf_str} |")

                lines += [""]

                # trechos recuperados
                lines += ["**Trechos recuperados:**", ""]
                for j, ctx in enumerate(contexts, 1):
                    tipo    = ctx.get("tipo_sigla", "?")
                    numero  = ctx.get("numero")
                    ano     = ctx.get("ano", "?")
                    num_str = str(int(numero)) if numero and str(numero) != "nan" else "s/n"
                    chunk_t = ctx.get("chunk_type", "?")
                    pid     = ctx.get("parent_id", "?")
                    trecho  = ctx.get("texto_parent", "*(vazio)*").strip()
                    if len(trecho) > 800:
                        trecho = trecho[:800] + "\n…*(truncado)*"

                    lines += [
                        f"<details>",
                        f"<summary>Fonte {j} — {tipo} {num_str}/{ano} [{chunk_t}] — {pid}</summary>",
                        "",
                        "```",
                        trecho,
                        "```",
                        "",
                        "</details>",
                        "",
                    ]
            else:
                lines += ["> ⚠️ Nenhuma fonte encontrada para esta query.", ""]

        # timing
        total = timing.get("total", 0)
        emb   = timing.get("embedding", 0)
        srch  = timing.get("hybrid_search", 0)
        rer   = timing.get("reranking", 0)
        gen   = timing.get("generation", 0)
        lines += [
            f"**Timing:** total `{total:.1f}s` | embedding `{emb:.1f}s` | "
            f"busca `{srch:.2f}s` | rerank `{rer:.1f}s` | geração `{gen:.1f}s`",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


# ── Pipeline de batch ─────────────────────────────────────────────────────────

def run_batch(questions: list[str], use_rerank: bool, debug: bool) -> list[dict]:
    from src._06_query_loader import query_pipeline  # carregado abaixo

    client = QdrantClient(path=str(QDRANT_PATH))
    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:80]}...")
        try:
            res = query_pipeline(
                question,
                client=client,
                debug=debug,
                use_rerank=use_rerank,
            )
            results.append(res)
            print(f"  ✓ {res['timing']['total']:.1f}s | "
                  f"{len(res['contexts'])} fontes | "
                  f"resposta: {len(res['answer'])} chars")
        except Exception as exc:
            logger.error(f"Erro na query {i}: {exc}")
            results.append({"question": question, "error": str(exc),
                             "answer": "", "contexts": [], "filters": {}, "timing": {}})

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Teste em lote do pipeline ANEEL RAG")
    parser.add_argument("--no-rerank", action="store_true", help="Pular reranking")
    parser.add_argument("--debug",     action="store_true", help="Scores e logs detalhados")
    args = parser.parse_args()

    use_rerank = not args.no_rerank

    print("=" * 60)
    print("ANEEL RAG — Batch Test")
    print(f"  Perguntas:  {len(QUESTIONS)}")
    print(f"  Reranking:  {'sim' if use_rerank else 'não'}")
    print("=" * 60)
    print("\nCarregando modelos (primeira query pode demorar)...")

    # importar o módulo de query via importlib para evitar side-effects no top-level
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "query_mod",
        Path(__file__).parent / "06_query.py",
    )
    query_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(query_mod)

    client = QdrantClient(path=str(QDRANT_PATH))
    results = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] {question[:80]}...")
        try:
            res = query_mod.query_pipeline(
                question,
                client=client,
                debug=args.debug,
                use_rerank=use_rerank,
            )
            results.append(res)

            t  = res["timing"]
            f  = res.get("filters", {})
            cr = res.get("critic")

            # Entidade detectada
            entity = f.get("entity_name")
            if entity:
                print(f"  Entidade: {entity}")

            # Timing por etapa
            embed  = t.get("embedding",    0)
            busca  = t.get("hybrid_search", 0)
            rerank = t.get("reranking",     0)
            gen    = t.get("generation",    0)
            valid  = t.get("validation",    0)
            total  = t.get("total",         0)
            timing_line = (
                f"  ── embed: {embed:.1f}s | busca: {busca:.1f}s | "
                f"rerank: {rerank:.1f}s | geração: {gen:.1f}s"
            )
            if valid:
                timing_line += f" | crítico: {valid:.1f}s"
            print(timing_line)

            # Resultado
            critic_str = ""
            if cr:
                if cr.get("valid"):
                    critic_str = f" | crítico: ✓ {cr.get('reason', 'ok')[:50]}"
                else:
                    critic_str = f" | crítico: ✗ REJEITOU"
            abstencao = res["answer"].startswith("Nao encontrei")
            status = "✗ SEM RESPOSTA" if abstencao else f"resposta: {len(res['answer'])} chars"
            print(f"  ✓ TOTAL {total:.1f}s | {len(res['contexts'])} fontes | {status}{critic_str}")
            print(f"  ╔ Resposta: {res['answer'][:500]}{'…' if len(res['answer']) > 500 else ''}")

        except Exception as exc:
            logger.error(f"Erro na query {i}: {exc}")
            results.append({
                "question": question,
                "error":    str(exc),
                "answer":   "",
                "contexts": [],
                "filters":  {},
                "timing":   {},
            })
            continue

    # salvar JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"batch_test_{ts}.json"
    md_path   = RESULTS_DIR / f"batch_test_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    md = build_markdown_report(results, {"rerank": use_rerank})
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    # resumo final
    ok    = sum(1 for r in results if not r.get("error") and r.get("answer"))
    total = sum(r.get("timing", {}).get("total", 0) for r in results)

    print("\n" + "=" * 60)
    print(f"  Concluído: {ok}/{len(QUESTIONS)} queries com resposta")
    print(f"  Tempo total: {total:.1f}s")
    print(f"  Relatório:  {md_path}")
    print(f"  JSON:       {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
