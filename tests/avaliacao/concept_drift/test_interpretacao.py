"""Testes de interpretacao de Concept Drift."""

from __future__ import annotations

import pytest

from monitoramento.avaliacao.concept_drift import (
    interpretacao,
)


def test_comparar_detecta_degradacao() -> None:
    resultado = interpretacao.comparar({"roc_auc": 0.70}, {"roc_auc": 0.65}, "roc_auc", 0.02)

    assert resultado["degradation_detected"] is True
    assert resultado["metric_delta"] == pytest.approx(-0.05)


def test_comparar_sem_degradacao_relevante() -> None:
    resultado = interpretacao.comparar({"roc_auc": 0.70}, {"roc_auc": 0.69}, "roc_auc", 0.02)

    assert resultado["degradation_detected"] is False


def test_montar_conclusoes_contrasta_data_e_concept_drift() -> None:
    referencia = {"roc_auc": 0.70, "accuracy": 0.70}
    alterado = {"roc_auc": 0.60, "accuracy": 0.60}
    comparacao = interpretacao.comparar(referencia, alterado, "roc_auc", 0.02)

    conclusoes = interpretacao.montar_conclusoes(
        referencia, alterado, comparacao, {"description": "recessao no segmento"}
    )
    texto = " ".join(conclusoes)

    assert "P(y|X)" in texto
    assert "P(X)" in texto
    assert "recessao no segmento" in texto
