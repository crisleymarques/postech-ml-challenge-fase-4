"""Construtores dos artefatos usados nos testes do painel."""

from __future__ import annotations

import json
from pathlib import Path


def escrever_execucao(
    runs_dir: Path,
    run_id: str,
    *,
    manifesto: dict | None = None,
    eventos: list[dict] | None = None,
) -> Path:
    """Cria um diretorio de execucao com manifesto e/ou events.jsonl."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    if manifesto is not None:
        (run_dir / "manifest.json").write_text(json.dumps(manifesto), encoding="utf-8")
    if eventos is not None:
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as arquivo:
            for evento in eventos:
                arquivo.write(json.dumps(evento) + "\n")
    return run_dir


def evento(run_id: str, stage: str, tipo: str) -> dict:
    """Evento minimo conforme o contrato comum."""
    return {"run_id": run_id, "stage": stage, "event": tipo, "level": "INFO"}
