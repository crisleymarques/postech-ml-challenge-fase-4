"""Teste de Kolmogorov-Smirnov para duas amostras."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def teste_ks(referencia: pd.Series, atual: pd.Series) -> tuple[float, float] | None:
    """Executa o teste de Kolmogorov-Smirnov de duas amostras."""
    esperado = pd.to_numeric(referencia, errors="coerce").dropna()
    observado = pd.to_numeric(atual, errors="coerce").dropna()
    if esperado.empty or observado.empty:
        return None
    resultado = stats.ks_2samp(esperado, observado)
    return float(resultado.statistic), float(resultado.pvalue)
