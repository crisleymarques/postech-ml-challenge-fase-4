"""Drift da integracao de alertas operacionais."""

from __future__ import annotations

from typing import Any

from .entradas import ID_ALERTAS_TOTAL, ID_DATASET_DRIFT, ID_PROPORCAO_DRIFT, ID_VARIAVEIS_DRIFT
from .regras import ORIGEM_AGREGADO

IDS_DRIFT_EXECUCAO = (ID_DATASET_DRIFT, ID_VARIAVEIS_DRIFT, ID_PROPORCAO_DRIFT)


def _ultimo_valor(medidas: list[dict[str, Any]], id_metrica: str, chave: str) -> Any:
    """Ultimo valor nao nulo de uma metrica, na ordem em que aparece."""
    valor = None
    for medida in medidas:
        if medida.get("id") == id_metrica and medida.get(chave) is not None:
            valor = medida.get(chave)
    return valor


def consolidar_drift(saude: dict[str, Any]) -> dict[str, Any]:
    """Resume uma unica execucao e preserva os resultados individuais de drift.

    Seleciona a maior proporcao, depois quantidade e run_id como desempates.
    Trata-se de destaque historico, nao de resultado mais recente. Nao infere
    dataset_drift quando a etapa (como a analise estatistica) nao o publicou.
    """
    por_execucao: list[dict[str, Any]] = []
    for execucao in saude.get("execucoes", []) or []:
        if not isinstance(execucao, dict):
            continue
        metricas = [item for item in execucao.get("metricas", []) or [] if isinstance(item, dict)]
        if any(item.get("id") in IDS_DRIFT_EXECUCAO for item in metricas):
            por_execucao.append(
                {
                    "run_id": execucao.get("run_id") or execucao.get("stage") or ORIGEM_AGREGADO,
                    "stage": execucao.get("stage"),
                    "dataset_drift": _ultimo_valor(metricas, ID_DATASET_DRIFT, "valor"),
                    "variaveis_com_drift": _ultimo_valor(metricas, ID_VARIAVEIS_DRIFT, "valor"),
                    "proporcao_drift": _ultimo_valor(metricas, ID_PROPORCAO_DRIFT, "valor"),
                }
            )

    por_execucao.sort(key=lambda item: item["run_id"])
    avaliadas = [
        item
        for item in por_execucao
        if any(
            item[chave] is not None
            for chave in (
                "dataset_drift",
                "variaveis_com_drift",
                "proporcao_drift",
            )
        )
    ]
    destaque = max(
        avaliadas,
        key=lambda item: (
            item["proporcao_drift"] if item["proporcao_drift"] is not None else -1,
            item["variaveis_com_drift"] if item["variaveis_com_drift"] is not None else -1,
            item["run_id"],
        ),
        default={},
    )

    agregados = [item for item in saude.get("agregados", []) or [] if isinstance(item, dict)]
    return {
        "dataset_drift": destaque.get("dataset_drift"),
        "variaveis_com_drift": destaque.get("variaveis_com_drift"),
        "proporcao_drift": destaque.get("proporcao_drift"),
        "alertas_total": _ultimo_valor(agregados, ID_ALERTAS_TOTAL, "valor"),
        "execucoes": [item["run_id"] for item in por_execucao],
        "execucao_resumo": destaque.get("run_id"),
        "criterio_resumo": "maior_proporcao_depois_quantidade_e_run_id",
        "por_execucao": por_execucao,
    }
