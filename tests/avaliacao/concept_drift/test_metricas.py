"""Testes de metricas de Concept Drift."""
from __future__ import annotations

import numpy as np
import pytest

from monitoramento.avaliacao.concept_drift import (
    config as configuracao,
)
from monitoramento.avaliacao.concept_drift import (
    metricas,
)


def test_calcular_metricas_retorna_valores() -> None:
    alvo = np.array([0, 0, 1, 1])
    predito = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.4, 0.6, 0.9])

    valores = metricas.calcular_metricas(alvo, predito, score, configuracao.METRICAS)

    assert set(valores) == set(configuracao.METRICAS)
    assert valores["accuracy"] == 0.5
    assert valores["roc_auc"] == pytest.approx(1.0)


def test_calcular_metricas_sem_classe_positiva_nao_calcula_auc() -> None:
    alvo = np.array([0, 0, 0, 0])
    predito = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.6, 0.9])

    valores = metricas.calcular_metricas(alvo, predito, score, ("roc_auc",))

    assert valores["roc_auc"] is None
