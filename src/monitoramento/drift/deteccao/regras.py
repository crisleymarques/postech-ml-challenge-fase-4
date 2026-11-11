"""Regras de decisao e resumo de drift independentes do Evidently."""

from __future__ import annotations

from typing import Any


def avaliar(
    estatistica: float, metodo: str | None, limiar_p: float, limiar_dist: float
) -> tuple[float, bool]:
    """Aplica a regra de deteccao e devolve o limiar usado e o resultado.

    Metodos de p-valor indicam drift quando a estatistica fica abaixo do limiar
    de p-valor; metodos de distancia, quando fica acima do limiar de distancia.
    """
    texto = str(metodo or "").lower()
    if "p_value" in texto or "p-value" in texto:
        return limiar_p, estatistica < limiar_p
    return limiar_dist, estatistica > limiar_dist


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume a deteccao com a contagem e a proporcao de variaveis com drift."""
    avaliaveis = [item for item in resultados if item["drift_detected"] is not None]
    com_drift = [item for item in avaliaveis if item["drift_detected"]]
    total = len(avaliaveis)
    proporcao = round(len(com_drift) / total, 6) if total else None
    return {
        "evaluated_columns": total,
        "drifted_columns": len(com_drift),
        "drift_share": proporcao,
        "dataset_drift": bool(total and len(com_drift) / total > 0.5),
        "drifted_features": [item["feature"] for item in com_drift],
    }
