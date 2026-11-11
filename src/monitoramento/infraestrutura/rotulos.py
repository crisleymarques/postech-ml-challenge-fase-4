"""Validacao do alvo binario antes de qualquer conversao destrutiva."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validar_rotulos(valores: pd.Series, *, erro: type[Exception]) -> np.ndarray:
    """Exige valores finitos e exatamente 0 ou 1; nunca trunca fracoes."""
    numericos = pd.to_numeric(valores, errors="coerce")
    if numericos.isna().any() or not np.isfinite(numericos.to_numpy()).all():
        raise erro("Os rotulos contem valores ausentes, nao numericos ou nao finitos.")
    if not numericos.isin([0, 1]).all():
        raise erro("Os rotulos devem pertencer ao dominio binario 0/1, sem fracoes.")
    return numericos.to_numpy().astype(int)
