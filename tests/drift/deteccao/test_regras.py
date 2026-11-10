"""Testes da deteccao de Data Drift por responsabilidade."""

from __future__ import annotations

from monitoramento.drift.deteccao import regras


def test_avaliar_aplica_regra_por_metodo() -> None:
    assert regras.avaliar(0.01, "K-S p_value", 0.05, 0.1) == (0.05, True)
    assert regras.avaliar(0.20, "K-S p_value", 0.05, 0.1) == (0.05, False)
    assert regras.avaliar(0.90, "Wasserstein distance", 0.05, 0.1) == (0.1, True)
    assert regras.avaliar(0.05, "Jensen-Shannon distance", 0.05, 0.1) == (0.1, False)


def test_resumir_calcula_proporcao() -> None:
    resultados = [
        {"feature": "a", "drift_detected": True},
        {"feature": "b", "drift_detected": False},
        {"feature": "c", "drift_detected": None},
    ]

    resumo = regras.resumir(resultados)

    assert resumo["evaluated_columns"] == 2
    assert resumo["drifted_columns"] == 1
    assert resumo["drift_share"] == 0.5
    assert resumo["dataset_drift"] is False
    assert resumo["drifted_features"] == ["a"]
