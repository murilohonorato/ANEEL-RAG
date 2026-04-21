"""
07_evaluate.py
--------------
Avaliação do pipeline com golden set + RAGAS.

Métricas:
- context_recall       — os chunks relevantes foram recuperados?
- context_precision    — os chunks recuperados são realmente relevantes?
- faithfulness         — a resposta se sustenta no contexto?
- answer_relevance     — a resposta responde a pergunta?

Golden set: tests/golden_set.json
    [{
        "question": "...",
        "expected_answer": "...",
        "relevant_doc_ids": ["ren20211000ti", ...]
    }]

Uso:
    python src/07_evaluate.py
    python src/07_evaluate.py --golden tests/golden_set.json
    python src/07_evaluate.py --output results/eval_001.json
"""

# CONFIG
GOLDEN_SET_PATH = "tests/golden_set.json"
OUTPUT_DIR      = "results"

# TODO: implementar com Claude Code
