"""Leitura da base e representacoes persistidas da simulacao."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .config import CONTRACT_VERSION, STAGE, ErroSimulacao


def carregar_base(caminho: Path) -> pd.DataFrame:
    """Le a base de origem mantendo os valores como texto."""
    if not caminho.is_file():
        raise ErroSimulacao(f"Base de origem nao encontrada: {caminho}")
    return pd.read_csv(caminho, dtype=str, na_filter=False)


def escrever_artefato(caminho: Path, base: pd.DataFrame) -> None:
    """Grava um artefato tabular em CSV preservando as colunas observadas."""
    base.to_csv(caminho, index=False, encoding="utf-8", mode="x")


def descritor_artefato(tipo: str, caminho: Path, run_id: str, base: pd.DataFrame) -> dict[str, Any]:
    """Descreve um artefato conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
        "row_count": int(len(base)),
        "columns": list(base.columns),
    }


def montar_manifesto(
    *,
    run_id: str,
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    batch_id: str,
    seed: int,
    entrada: dict[str, Any] | None,
    saidas: list[dict[str, Any]],
    periodo: dict[str, Any] | None,
    descritas: list[dict[str, Any]],
    status: str = "success",
    error_code: str | None = None,
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor do cenario simulado."""
    variaveis = sorted(descritas, key=lambda item: item["column"])
    transformacoes = (
        [f"deslocamento temporal de {periodo['column']} em {periodo['shift_months']} meses"]
        if periodo
        else []
    )
    transformacoes.extend(
        f"{item['operation']} em {item['column']} (parametro {item['parameter']})"
        for item in variaveis
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": status,
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": [entrada] if entrada else [],
        "outputs": saidas,
        "dataset_id": config.get("dataset_id"),
        "schema_version": config.get("schema_version"),
        "model_id": None,
        "reference_id": None,
        "scenario_id": config.get("scenario_id"),
        "batch_id": batch_id,
        "seed": seed,
        "periodo": periodo,
        "variaveis_alteradas": [item["column"] for item in variaveis],
        "transformacoes": transformacoes,
        "parametros": {
            "jitter": float(config["drift"].get("jitter", 0.0)),
            "shift_months": int(periodo["shift_months"]) if periodo else None,
            "variables": {item["column"]: item for item in variaveis},
        },
        "error_code": error_code,
    }


def montar_medidas(
    saidas: list[dict[str, Any]], descritas: list[dict[str, Any]], shift: int
) -> list:
    """Converte a simulacao gerada em medidas do contrato de metricas."""
    linhas = sum(item["row_count"] for item in saidas if item["artifact_type"] == "features")
    return [
        {
            "name": "linhas_geradas",
            "value": int(linhas),
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "variaveis_alteradas",
            "value": len(descritas),
            "unit": "variaveis",
            "availability": "available",
        },
        {
            "name": "deslocamento_meses",
            "value": int(shift),
            "unit": "meses",
            "availability": "available",
        },
    ]
