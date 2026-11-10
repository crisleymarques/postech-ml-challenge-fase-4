"""Publicacao e representacoes dos resultados da predicao."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .config import CONTRACT_VERSION, STAGE


def escrever_predicoes(caminho: Path, base: pd.DataFrame) -> None:
    """Grava as predicoes em CSV."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(caminho, index=False, encoding="utf-8")


def descritor_artefato(tipo: str, caminho: Path, run_id: str, base: pd.DataFrame) -> dict[str, Any]:
    """Descreve um artefato tabular conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
        "row_count": int(len(base)),
        "columns": list(base.columns),
    }


def montar_medidas(predicoes: pd.DataFrame, score_disponivel: bool) -> list[dict[str, Any]]:
    """Converte as predicoes geradas em medidas do contrato de metricas."""
    total = int(len(predicoes))
    positivos = int(predicoes["prediction"].sum()) if total else 0
    if score_disponivel and total:
        score_medio: float | None = round(float(predicoes["score"].mean()), 6)
        disponibilidade_score = "available"
    else:
        score_medio = None
        disponibilidade_score = "not_available"
    return [
        {
            "name": "predicoes_geradas",
            "value": total,
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "taxa_positivos",
            "value": round(positivos / total, 6) if total else None,
            "unit": "proporcao",
            "availability": "available" if total else "not_available",
        },
        {
            "name": "score_medio",
            "value": score_medio,
            "unit": "probabilidade",
            "availability": disponibilidade_score,
        },
    ]


def montar_manifesto(
    *,
    run_id: str,
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    manifesto_lote: dict[str, Any],
    modelo_id: str | None,
    referencia_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
    entradas: list[dict[str, Any]],
    saidas: list[dict[str, Any]],
    score: dict[str, Any],
    linhas: int,
    status: str = "success",
    error_code: str | None = None,
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor das predicoes."""
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": status,
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": entradas,
        "outputs": saidas,
        "dataset_id": config["dataset_id"],
        "schema_version": manifesto_lote.get("schema_version"),
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "row_count": int(linhas),
        "score": score,
        "error_code": error_code,
    }



def publicar_manifesto(caminho: Path, manifesto: dict[str, Any]) -> None:
    """Publica o manifesto completo atomicamente, apos os demais artefatos."""
    from monitoramento.infraestrutura.arquivos import escrever_json

    with NamedTemporaryFile(dir=caminho.parent, suffix=".tmp", delete=False) as arquivo:
        temporario = Path(arquivo.name)
    try:
        escrever_json(temporario, manifesto)
        os.replace(temporario, caminho)
    finally:
        temporario.unlink(missing_ok=True)
