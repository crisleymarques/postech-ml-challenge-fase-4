"""Eventos de execucao em UTC no contrato comum de observabilidade."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def agora_iso() -> str:
    """Retorna o instante atual em ISO 8601 UTC com sufixo Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def registrar_evento(
    eventos: list[dict[str, Any]],
    run_id: str,
    dataset_id: str,
    scenario_id: str,
    batch_id: str,
    evento: str,
    nivel: str,
    error_code: str | None = None,
    *,
    stage: str,
    model_id: str | None = None,
    reference_id: str | None = None,
) -> None:
    """Acrescenta um evento ao contrato comum de observabilidade."""
    eventos.append(
        {
            "timestamp": agora_iso(),
            "run_id": run_id,
            "stage": stage,
            "event": evento,
            "level": nivel,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "reference_id": reference_id,
            "scenario_id": scenario_id,
            "batch_id": batch_id,
            "error_code": error_code,
        }
    )


def escrever_eventos(caminho: Path, eventos: list[dict[str, Any]]) -> None:
    """Grava os eventos em JSON Lines, um objeto por linha."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        for evento in eventos:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
