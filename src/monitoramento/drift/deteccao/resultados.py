"""Documentos e metricas especificos da deteccao de Data Drift."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .adaptador_evidently import VERSAO_EVIDENTLY
from .config import CONTRACT_VERSION, PRESET, STAGE


def montar_medidas(resumo: dict[str, Any]) -> list[dict[str, Any]]:
    """Converte o resumo da deteccao em medidas do contrato de metricas."""
    disponibilidade = "available" if resumo["evaluated_columns"] else "not_available"
    return [
        {
            "name": "variaveis_avaliadas",
            "value": resumo["evaluated_columns"],
            "unit": "variaveis",
            "availability": disponibilidade,
        },
        {
            "name": "variaveis_com_drift",
            "value": resumo["drifted_columns"],
            "unit": "variaveis",
            "availability": disponibilidade,
        },
        {
            "name": "proporcao_drift",
            "value": resumo["drift_share"],
            "unit": "proporcao",
            "availability": "available" if resumo["drift_share"] is not None else "not_available",
        },
    ]


def descritor_artefato(tipo: str, caminho: Path, run_id: str) -> dict[str, Any]:
    """Descreve um artefato de saida nao tabular conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }


def montar_documento(
    *,
    run_id: str,
    config: dict[str, Any],
    modelo_id: str | None,
    referencia_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
    caminho_ref: Path,
    caminho_lote: Path,
    resultados: list[dict[str, Any]],
    resumo: dict[str, Any],
    linhas_ref: int,
    linhas_lote: int,
    alteradas: list[str],
) -> dict[str, Any]:
    """Reune o documento de drift com ferramenta, metodo, amostras e resultados."""
    detectadas = [variavel for variavel in alteradas if variavel in resumo["drifted_features"]]
    nao_detectadas = [
        variavel for variavel in alteradas if variavel not in resumo["drifted_features"]
    ]
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "dataset_id": config["dataset_id"],
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "tool": {"name": "evidently", "version": VERSAO_EVIDENTLY},
        "method": {
            "name": config["method"]["name"],
            "preset": config["method"].get("preset", PRESET),
            "p_value_threshold": float(config["method"]["p_value_threshold"]),
            "distance_threshold": float(config["method"]["distance_threshold"]),
        },
        "samples": {
            "reference": {"path": caminho_ref.as_posix(), "row_count": int(linhas_ref)},
            "current": {"path": caminho_lote.as_posix(), "row_count": int(linhas_lote)},
        },
        "features": resultados,
        "summary": resumo,
        "changed_variables": {
            "configured": list(alteradas),
            "detected": detectadas,
            "not_detected": nao_detectadas,
        },
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
    resumo: dict[str, Any],
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor da deteccao."""
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
        "tool": {"name": "evidently", "version": VERSAO_EVIDENTLY},
        "drift": resumo,
        "error_code": None,
    }
