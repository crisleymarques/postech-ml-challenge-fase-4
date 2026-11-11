"""Population Stability Index numerico e categorico."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-6


def indice_psi(esperado: np.ndarray, observado: np.ndarray) -> float:
    """Calcula o PSI entre duas distribuicoes de proporcao alinhadas."""
    esperado = np.where(np.asarray(esperado, dtype=float) <= 0, EPSILON, esperado)
    observado = np.where(np.asarray(observado, dtype=float) <= 0, EPSILON, observado)
    return float(np.sum((observado - esperado) * np.log(observado / esperado)))


def psi_numerico(referencia: pd.Series, atual: pd.Series, bins: int) -> float | None:
    """Calcula o PSI de uma variavel numerica usando faixas por quantis."""
    esperado = pd.to_numeric(referencia, errors="coerce").dropna()
    observado = pd.to_numeric(atual, errors="coerce").dropna()
    if esperado.empty or observado.empty:
        return None

    limites = np.unique(np.quantile(esperado, np.linspace(0, 1, bins + 1)))
    if len(limites) < 3:
        return None
    limites = limites.astype(float)
    limites[0] = -np.inf
    limites[-1] = np.inf

    contagem_ref, _ = np.histogram(esperado, bins=limites)
    contagem_atual, _ = np.histogram(observado, bins=limites)
    prop_ref = contagem_ref / contagem_ref.sum()
    prop_atual = contagem_atual / contagem_atual.sum()
    return indice_psi(prop_ref, prop_atual)


def psi_categorico(referencia: pd.Series, atual: pd.Series) -> float | None:
    """Calcula o PSI de uma variavel categorica pelas proporcoes por categoria."""
    esperado = referencia.astype("object").fillna("<ausente>")
    observado = atual.astype("object").fillna("<ausente>")
    if esperado.empty or observado.empty:
        return None

    categorias = sorted(set(esperado.unique()) | set(observado.unique()))
    prop_ref = esperado.value_counts(normalize=True)
    prop_atual = observado.value_counts(normalize=True)
    vetor_ref = np.array([prop_ref.get(categoria, 0.0) for categoria in categorias])
    vetor_atual = np.array([prop_atual.get(categoria, 0.0) for categoria in categorias])
    return indice_psi(vetor_ref, vetor_atual)


def faixa_psi(valor: float, moderado: float, limiar: float) -> str:
    """Classifica o PSI em insignificante, moderada ou significativa."""
    if valor >= limiar:
        return "significativa"
    if valor >= moderado:
        return "moderada"
    return "insignificante"
