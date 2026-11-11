"""Testes de psi da analise estatistica."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoramento.drift.analise import psi

from .apoio import (
    QUANTIDADE,
    SEMENTE,
)


def test_indice_psi_zero_para_distribuicoes_iguais() -> None:
    igual = np.array([0.5, 0.5])

    assert psi.indice_psi(igual, igual) == pytest.approx(0.0)


def test_indice_psi_positivo_para_distribuicoes_diferentes() -> None:
    esperado = np.array([0.5, 0.5])
    observado = np.array([0.9, 0.1])

    assert psi.indice_psi(esperado, observado) > 0.25


def test_psi_numerico_alto_com_deslocamento() -> None:
    rng = np.random.default_rng(SEMENTE)
    referencia = pd.Series(rng.normal(60000, 5000, QUANTIDADE))
    atual = pd.Series(rng.normal(120000, 5000, QUANTIDADE))

    assert psi.psi_numerico(referencia, atual, 10) > 0.25


def test_psi_categorico_alto_com_proporcoes_diferentes() -> None:
    referencia = pd.Series(["RENT"] * 80 + ["OWN"] * 20)
    atual = pd.Series(["RENT"] * 10 + ["OWN"] * 90)

    assert psi.psi_categorico(referencia, atual) > 0.25


def test_faixa_psi_classifica() -> None:
    assert psi.faixa_psi(0.05, 0.1, 0.25) == "insignificante"
    assert psi.faixa_psi(0.15, 0.1, 0.25) == "moderada"
    assert psi.faixa_psi(0.30, 0.1, 0.25) == "significativa"
