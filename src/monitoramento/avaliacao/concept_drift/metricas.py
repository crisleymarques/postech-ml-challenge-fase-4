"""Metricas de classificacao dos dois conceitos."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def calcular_metricas(
    alvo: np.ndarray,
    predito: np.ndarray,
    score: np.ndarray | None,
    metricas: tuple[str, ...],
) -> dict[str, float | None]:
    """Calcula as metricas de classificacao de um conjunto de rotulos."""
    resultado: dict[str, float | None] = {}
    for metrica in metricas:
        if metrica == "roc_auc":
            if score is not None and len(np.unique(alvo)) > 1:
                resultado["roc_auc"] = round(float(roc_auc_score(alvo, score)), 6)
            else:
                resultado["roc_auc"] = None
        elif metrica == "accuracy":
            resultado["accuracy"] = round(float(accuracy_score(alvo, predito)), 6)
        elif metrica == "precision":
            resultado["precision"] = round(
                float(precision_score(alvo, predito, zero_division=0)), 6
            )
        elif metrica == "recall":
            resultado["recall"] = round(float(recall_score(alvo, predito, zero_division=0)), 6)
        elif metrica == "f1":
            resultado["f1"] = round(float(f1_score(alvo, predito, zero_division=0)), 6)
    return resultado
