"""Testes de kolmogorov_smirnov da analise estatistica."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoramento.drift.analise import kolmogorov_smirnov as ks

from .apoio import (
    QUANTIDADE,
    SEMENTE,
)


def test_teste_ks_identico_tem_p_valor_alto() -> None:
    rng = np.random.default_rng(SEMENTE)
    amostra = pd.Series(rng.normal(0, 1, QUANTIDADE))

    estatistica, pvalor = ks.teste_ks(amostra, amostra)

    assert estatistica == pytest.approx(0.0)
    assert pvalor > 0.05


def test_teste_ks_detecta_amostras_deslocadas() -> None:
    rng = np.random.default_rng(SEMENTE)
    referencia = pd.Series(rng.normal(0, 1, QUANTIDADE))
    atual = pd.Series(rng.normal(5, 1, QUANTIDADE))

    estatistica, pvalor = ks.teste_ks(referencia, atual)

    assert estatistica > 0.5
    assert pvalor < 0.05
