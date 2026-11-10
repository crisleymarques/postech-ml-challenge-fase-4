"""Documentos e medidas da analise estatistica."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import scipy

from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .config import CONTRACT_VERSION, CRITERIO_KS, CRITERIO_PSI, STAGE
from .interpretacao import montar_conclusoes


def montar_medidas(resumo: dict[str, Any], alteradas: list[str]) -> list[dict[str, Any]]:
    """Converte o resumo da analise em medidas do contrato de metricas."""
    confirmadas = [coluna for coluna in alteradas if coluna in resumo["changed_features"]]
    disponibilidade = "available" if resumo["evaluated_columns"] else "not_available"
    return [
        {
            "name": "variaveis_avaliadas",
            "value": resumo["evaluated_columns"],
            "unit": "variaveis",
            "availability": disponibilidade,
        },
        {
            "name": "variaveis_com_mudanca",
            "value": resumo["changed_columns"],
            "unit": "variaveis",
            "availability": disponibilidade,
        },
        {
            "name": "proporcao_mudanca",
            "value": resumo["change_share"],
            "unit": "proporcao",
            "availability": "available" if resumo["change_share"] is not None else "not_available",
        },
        {
            "name": "psi_maximo",
            "value": resumo["psi_maximo"],
            "unit": "indice",
            "availability": "available" if resumo["psi_maximo"] is not None else "not_available",
        },
        {
            "name": "variaveis_alteradas_confirmadas",
            "value": len(confirmadas),
            "unit": "variaveis",
            "availability": "available" if alteradas else "not_applicable",
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
    """Reune o documento da analise com criterios, resultados e interpretacao."""
    confirmadas = [coluna for coluna in alteradas if coluna in resumo["changed_features"]]
    nao_confirmadas = [coluna for coluna in alteradas if coluna not in resumo["changed_features"]]
    metodo = config["method"]
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "dataset_id": config["dataset_id"],
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "tool": {"name": "scipy", "version": scipy.__version__},
        "method": {
            "name": metodo["name"],
            "bins": int(metodo["bins"]),
            "psi_moderate": float(metodo["psi_moderate"]),
            "psi_threshold": float(metodo["psi_threshold"]),
            "ks_p_value_threshold": float(metodo["ks_p_value_threshold"]),
            "ks_statistic_threshold": float(metodo["ks_statistic_threshold"]),
        },
        "criteria": {"psi": CRITERIO_PSI, "ks": CRITERIO_KS},
        "samples": {
            "reference": {"path": caminho_ref.as_posix(), "row_count": int(linhas_ref)},
            "current": {"path": caminho_lote.as_posix(), "row_count": int(linhas_lote)},
        },
        "features": resultados,
        "summary": resumo,
        "changed_variables": {
            "configured": list(alteradas),
            "confirmed": confirmadas,
            "not_confirmed": nao_confirmadas,
        },
        "interpretation": {
            "conclusions": montar_conclusoes(
                resumo,
                alteradas,
                psi_limiar=float(metodo["psi_threshold"]),
                ks_p=float(metodo["ks_p_value_threshold"]),
            )
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
    """Reune o manifesto comum do produtor da analise."""
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
        "tool": {"name": "scipy", "version": scipy.__version__},
        "analysis": resumo,
        "error_code": None,
    }
