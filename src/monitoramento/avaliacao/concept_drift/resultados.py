"""Documentos e medidas da avaliacao de Concept Drift."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sklearn

from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .config import CONTRACT_VERSION, STAGE
from .interpretacao import montar_conclusoes


def montar_medidas(
    comparacao: dict[str, Any],
    referencia: dict[str, float | None],
    alterado: dict[str, float | None],
) -> list[dict[str, Any]]:
    """Converte a comparacao em medidas do contrato de metricas."""
    metrica = comparacao["metric"]
    valor_ref = referencia.get(metrica)
    valor_alt = alterado.get(metrica)
    disponibilidade = (
        "available" if valor_ref is not None and valor_alt is not None else "not_available"
    )
    return [
        {
            "name": f"{metrica}_referencia",
            "value": valor_ref,
            "unit": "metrica",
            "availability": disponibilidade,
        },
        {
            "name": f"{metrica}_concept_drift",
            "value": valor_alt,
            "unit": "metrica",
            "availability": disponibilidade,
        },
        {
            "name": f"variacao_{metrica}",
            "value": comparacao["metric_delta"],
            "unit": "variacao",
            "availability": disponibilidade,
        },
        {
            "name": "degradacao_detectada",
            "value": 1 if comparacao["degradation_detected"] else 0,
            "unit": "indicador",
            "availability": "available",
        },
    ]


def montar_documento(
    *,
    run_id: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
    modelo_id: str | None,
    referencia_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
    caminho_ref: Path,
    caminho_alt: Path,
    linhas: int,
    referencia: dict[str, float | None],
    alterado: dict[str, float | None],
    comparacao: dict[str, Any],
    conceito: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reune o documento da avaliacao com metricas, comparacao e interpretacao."""
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "dataset_id": config["dataset_id"],
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "tool": {"name": "scikit-learn", "version": sklearn.__version__},
        "metrics": list(config["metrics"]),
        "positive_class": metadata.get("target", {}).get("positive_class"),
        "prediction_threshold": float(metadata.get("prediction_threshold", 0.5)),
        "contrast": {
            "reference": {
                "concept": "conceito original (rotulos de referencia)",
                "labels_path": caminho_ref.as_posix(),
                "row_count": int(linhas),
                "metrics": referencia,
            },
            "concept_drift": {
                "concept": "cenario de Concept Drift",
                "labels_path": caminho_alt.as_posix(),
                "row_count": int(linhas),
                "metrics": alterado,
            },
        },
        "comparison": comparacao,
        "concept_change": conceito,
        "interpretation": {
            "conclusions": montar_conclusoes(referencia, alterado, comparacao, conceito)
        },
    }


def descritor_artefato(tipo: str, caminho: Path, run_id: str) -> dict[str, Any]:
    """Descreve um artefato de saida nao tabular conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }


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
    comparacao: dict[str, Any],
    linhas: int,
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor da avaliacao."""
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "success",
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
        "tool": {"name": "scikit-learn", "version": sklearn.__version__},
        "comparison": comparacao,
        "row_count": int(linhas),
        "error_code": None,
    }
