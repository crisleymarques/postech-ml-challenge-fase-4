"""Testes e apoio de apoio."""

from __future__ import annotations

import json
from pathlib import Path


def escrever_execucao(
    runs_dir: Path,
    run_id: str,
    eventos: list[dict],
    manifesto: dict | None = None,
) -> Path:
    """Cria um diretorio de execucao com events.jsonl e manifesto opcional."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as arquivo:
        for evento in eventos:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
    if manifesto is not None:
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as arquivo:
            json.dump(manifesto, arquivo, ensure_ascii=False, indent=2)
    return run_dir


def evento(
    *,
    run_id: str,
    stage: str,
    tipo: str,
    nivel: str = "INFO",
    error_code: str | None = None,
    timestamp: str = "2026-11-09T00:00:00.000000Z",
    scenario_id: str | None = None,
    batch_id: str | None = None,
) -> dict:
    """Evento minimo conforme o contrato comum de observabilidade."""
    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "stage": stage,
        "event": tipo,
        "level": nivel,
        "dataset_id": "teste",
        "model_id": None,
        "reference_id": None,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "error_code": error_code,
    }
