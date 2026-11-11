"""Aplicacao do pipeline e construcao das predicoes alinhadas ao lote."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import IDENTIFICADOR, ErroPredicao

COLUNAS_SAIDA = (IDENTIFICADOR, "prediction", "score")


def prever(
    pipeline: Any, metadata: dict[str, Any], lote: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    """Aplica o pipeline do treino e monta a saida de predicoes.

    Devolve o quadro de predicoes e se o score esta disponivel.
    """
    colunas = list(metadata.get("features", []))
    if not colunas:
        raise ErroPredicao("Metadados do modelo sem lista de features.")
    if IDENTIFICADOR not in lote.columns:
        raise ErroPredicao(f"Identificador ausente no lote: {IDENTIFICADOR}")

    alvo = metadata.get("target", {}).get("column")
    if alvo and alvo in lote.columns:
        raise ErroPredicao(f"Lote de producao nao deve conter o alvo: {alvo}")

    ausentes = [coluna for coluna in colunas if coluna not in lote.columns]
    if ausentes:
        raise ErroPredicao(f"Features ausentes no lote: {ausentes}")

    entrada = lote[colunas]
    rotulos = pipeline.predict(entrada)

    if hasattr(pipeline, "predict_proba"):
        positivo = metadata.get("target", {}).get("positive_value", 1)
        classes = list(getattr(pipeline, "classes_", []))
        indice = classes.index(positivo) if positivo in classes else 1
        score = np.asarray(pipeline.predict_proba(entrada))[:, indice]
        score_disponivel = True
    else:
        score = np.full(len(entrada), np.nan)
        score_disponivel = False

    predicoes = pd.DataFrame(
        {
            IDENTIFICADOR: lote[IDENTIFICADOR].to_numpy(),
            "prediction": np.asarray(rotulos).astype(int),
            "score": np.round(score, 6),
        }
    )
    if not predicoes[IDENTIFICADOR].is_unique:
        raise ErroPredicao("O lote possui identificadores repetidos.")
    return predicoes, score_disponivel
