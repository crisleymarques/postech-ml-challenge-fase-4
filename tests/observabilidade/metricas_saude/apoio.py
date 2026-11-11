"""Construtores de metricas e artefatos dos testes de saude."""

from __future__ import annotations

import json
from pathlib import Path

CATALOGO_REAL = Path(__file__).resolve().parents[3] / "configs/metricas_saude.json"


def metrica(**campos: object) -> dict:
    """Metrica minima conforme o catalogo, sobrescrita pelos campos informados."""
    base = {
        "id": "categoria.metrica",
        "categoria": "pipeline",
        "nome": "Metrica",
        "unidade": "unidade",
        "escopo": "execucao",
        "etapas": None,
        "coleta": "fonte",
        "criterio": "criterio",
        "direcao": None,
        "limiar_degracao": None,
        "limiar_falha": None,
    }
    base.update(campos)
    return base


def escrever_run(
    runs_dir: Path,
    run_id: str,
    *,
    manifesto: dict | None = None,
    metrics: dict | None = None,
    quality: dict | None = None,
    drift: dict | None = None,
    analise: dict | None = None,
    performance: dict | None = None,
    eventos: list[dict] | None = None,
) -> Path:
    """Cria um diretorio de execucao com os artefatos informados."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    for nome, conteudo in (
        ("manifest.json", manifesto),
        ("metrics.json", metrics),
        ("quality.json", quality),
        ("drift.json", drift),
        ("analise.json", analise),
        ("performance.json", performance),
    ):
        if conteudo is not None:
            (run_dir / nome).write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")
    if eventos is not None:
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as arquivo:
            for evento in eventos:
                arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
    return run_dir


def evento_drift(run_id: str, stage: str) -> dict:
    """Evento de alerta de drift conforme o contrato comum."""
    return {
        "timestamp": "2026-11-09T02:50:49.872109Z",
        "run_id": run_id,
        "stage": stage,
        "event": "drift_alert",
        "level": "WARNING",
        "dataset_id": "lending_club_100k",
        "model_id": "baseline-logreg-820dbb61",
        "reference_id": "ref-820dbb61",
        "scenario_id": "data-drift",
        "batch_id": "batch-f201",
        "error_code": None,
    }
