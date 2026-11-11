"""Execucao, extracao e HTML do Evidently."""

from __future__ import annotations

import math
from typing import Any

import evidently
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from .regras import avaliar

TIPO_DRIFT = "evidently:metric_v2:ValueDrift"
VERSAO_EVIDENTLY = evidently.__version__


def executar_evidently(referencia: pd.DataFrame, atual: pd.DataFrame) -> Any:
    """Executa o preset de Data Drift do Evidently sobre os dois conjuntos."""
    relatorio = Report([DataDriftPreset()], include_tests=False)
    return relatorio.run(reference_data=referencia, current_data=atual)


def extrair_estatisticas(
    snapshot: Any, limiar_p: float, limiar_dist: float
) -> dict[str, dict[str, Any]]:
    """Extrai o resultado de drift por variavel a partir do snapshot do Evidently."""
    encontrados: dict[str, dict[str, Any]] = {}
    for item in snapshot.dict().get("metrics", []):
        config = item.get("config", {})
        if config.get("type") != TIPO_DRIFT:
            continue
        coluna = config.get("column")
        metodo = config.get("method")
        valor = item.get("value")
        if valor is None or not math.isfinite(float(valor)):
            encontrados[coluna] = {
                "feature": coluna,
                "method": metodo,
                "statistic": None,
                "threshold": None,
                "drift_detected": None,
                "reason": "variavel nao avaliavel pelo Evidently",
            }
            continue
        estatistica = float(valor)
        limiar, drift = avaliar(estatistica, metodo, limiar_p, limiar_dist)
        encontrados[coluna] = {
            "feature": coluna,
            "method": metodo,
            "statistic": round(estatistica, 6),
            "threshold": limiar,
            "drift_detected": drift,
        }
    return encontrados


def montar_resultados(
    snapshot: Any, colunas: list[str], limiar_p: float, limiar_dist: float
) -> list[dict[str, Any]]:
    """Monta o resultado por variavel preservando a ordem das features."""
    encontrados = extrair_estatisticas(snapshot, limiar_p, limiar_dist)
    resultados: list[dict[str, Any]] = []
    for coluna in colunas:
        if coluna in encontrados:
            resultados.append(encontrados[coluna])
        else:
            resultados.append(
                {
                    "feature": coluna,
                    "method": None,
                    "statistic": None,
                    "threshold": None,
                    "drift_detected": None,
                    "reason": "sem resultado de drift do Evidently para a variavel",
                }
            )
    return resultados


def comparar_features(
    referencia: pd.DataFrame,
    atual: pd.DataFrame,
    colunas: list[str],
    limiar_p: float,
    limiar_dist: float,
) -> list[dict[str, Any]]:
    """Calcula somente colunas avaliaveis e preserva a cobertura completa."""
    indisponiveis = {}
    for coluna in colunas:
        vazias = [
            papel
            for papel, base in (("referencia", referencia), ("producao", atual))
            if base[coluna].dropna().empty
        ]
        if vazias:
            indisponiveis[coluna] = {
                "feature": coluna,
                "method": None,
                "statistic": None,
                "threshold": None,
                "drift_detected": None,
                "reason": f"coluna sem valores observados: {', '.join(vazias)}",
            }
    avaliaveis = [coluna for coluna in colunas if coluna not in indisponiveis]
    encontrados = {}
    if avaliaveis:
        snapshot = executar_evidently(referencia[avaliaveis], atual[avaliaveis])
        encontrados = {
            item["feature"]: item
            for item in montar_resultados(snapshot, avaliaveis, limiar_p, limiar_dist)
        }
    return [indisponiveis.get(coluna) or encontrados[coluna] for coluna in colunas]
