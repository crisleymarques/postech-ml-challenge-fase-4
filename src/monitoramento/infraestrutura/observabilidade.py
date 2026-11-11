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


def status_publicado(manifesto: dict[str, Any], eventos: list[dict[str, Any]]) -> str:
    """Sucesso exige publicacao terminal e eventos coerentes com a execucao.

    Eventos de falha sem manifesto continuam disponiveis para diagnostico.
    Um completed isolado nao confirma a publicacao dos artefatos.
    """
    terminais = {
        "completed": "success",
        "rejected": "rejected",
        "failed": "failed",
        "skipped": "skipped",
    }
    terminal = next(
        (e for e in reversed(eventos) if isinstance(e, dict) and e.get("event") in terminais),
        {},
    )
    declarado = manifesto.get("status")
    observado = terminais.get(terminal.get("event"), "incomplete")
    if declarado != "success" and observado != "success":
        return declarado if declarado in terminais.values() else observado
    if (
        declarado != "success"
        or str(manifesto.get("contract_version", "")).split(".")[0] != "1"
        or not manifesto.get("run_id")
        or not manifesto.get("stage")
        or observado != "success"
        or any(terminal.get(k) != manifesto[k] for k in ("run_id", "stage"))
    ):
        return "incomplete"
    drift = manifesto.get("drift")
    if (
        isinstance(drift, dict)
        and drift.get("dataset_drift")
        and not any(
            isinstance(e, dict)
            and e.get("event") == "drift_alert"
            and all(e.get(k) == manifesto[k] for k in ("run_id", "stage"))
            for e in eventos
        )
    ):
        return "incomplete"
    return "success"
