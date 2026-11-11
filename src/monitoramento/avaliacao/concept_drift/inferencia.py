"""Inferencia da avaliacao com o limiar declarado nos metadados."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import IDENTIFICADOR, ErroAvaliacao


def prever(
    pipeline: Any, metadata: dict[str, Any], lote: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray | None]:
    """Aplica o pipeline do treino e devolve a predicao e o score."""
    colunas = list(metadata.get("features", []))
    if not colunas:
        raise ErroAvaliacao("Metadados do modelo sem lista de features.")
    if IDENTIFICADOR not in lote.columns:
        raise ErroAvaliacao(f"Identificador ausente no lote: {IDENTIFICADOR}")

    alvo = metadata.get("target", {}).get("column")
    if alvo and alvo in lote.columns:
        raise ErroAvaliacao(f"Lote de producao nao deve conter o alvo: {alvo}")

    ausentes = [coluna for coluna in colunas if coluna not in lote.columns]
    if ausentes:
        raise ErroAvaliacao(f"Features ausentes no lote: {ausentes}")

    entrada = lote[colunas]

    if hasattr(pipeline, "predict_proba"):
        positivo = metadata.get("target", {}).get("positive_value", 1)
        classes = list(getattr(pipeline, "classes_", []))
        indice = classes.index(positivo) if positivo in classes else 1
        score = np.asarray(pipeline.predict_proba(entrada))[:, indice]
        limiar = float(metadata.get("prediction_threshold", 0.5))
        predito = (score >= limiar).astype(int)
        return predito, score

    predito = np.asarray(pipeline.predict(entrada)).astype(int)
    return predito, None
