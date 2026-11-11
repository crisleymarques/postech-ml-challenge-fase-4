"""Leitura e representacoes dos artefatos de Concept Drift."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .config import CONTRACT_VERSION, OPERACAO, STAGE, ErroSimulacao


def carregar_base(caminho: Path) -> pd.DataFrame:
    """Le a base de origem mantendo os valores como texto."""
    if not caminho.is_file():
        raise ErroSimulacao(f"Base de origem nao encontrada: {caminho}")
    return pd.read_csv(caminho, dtype=str, na_filter=False)


def escrever_artefato(caminho: Path, base: pd.DataFrame) -> None:
    """Grava um artefato tabular em CSV preservando as colunas observadas."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=caminho.parent, suffix=".tmp", delete=False) as arquivo:
        temporario = Path(arquivo.name)
    try:
        base.to_csv(temporario, index=False, encoding="utf-8")
        os.replace(temporario, caminho)
    finally:
        temporario.unlink(missing_ok=True)


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


def montar_medidas(descricao: dict[str, Any], linhas: int) -> list[dict[str, Any]]:
    """Converte o cenario gerado em medidas do contrato de metricas."""
    return [
        {
            "name": "linhas_geradas",
            "value": int(linhas),
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "registros_no_segmento",
            "value": descricao["segment_rows"],
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "rotulos_alterados",
            "value": descricao["flipped_rows"],
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "taxa_positivos_antes",
            "value": descricao["positive_rate_before"],
            "unit": "proporcao",
            "availability": "available",
        },
        {
            "name": "taxa_positivos_depois",
            "value": descricao["positive_rate_after"],
            "unit": "proporcao",
            "availability": "available",
        },
    ]


def montar_manifesto(
    *,
    run_id: str,
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    batch_id: str,
    seed: int,
    entrada: dict[str, Any],
    saidas: list[dict[str, Any]],
    descricao: dict[str, Any],
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor do cenario de Concept Drift."""
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "success",
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": [entrada],
        "outputs": saidas,
        "dataset_id": config.get("dataset_id"),
        "schema_version": config.get("schema_version"),
        "model_id": None,
        "reference_id": None,
        "scenario_id": config.get("scenario_id"),
        "batch_id": batch_id,
        "seed": seed,
        "drift_type": "concept",
        "features_unchanged": True,
        "concept_change": descricao,
        "transformacoes": [
            f"mudanca de conceito em P(y|X): {OPERACAO} no segmento "
            f"{descricao['segment']['column']} {descricao['segment']['operator']} "
            f"{descricao['segment']['value']}"
        ],
        "error_code": None,
    }
