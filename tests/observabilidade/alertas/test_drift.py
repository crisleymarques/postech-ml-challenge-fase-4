"""Testes de drift dos alertas."""

from __future__ import annotations

from monitoramento.observabilidade.alertas import drift as al

from .apoio import medida, saude


def test_consolidar_drift() -> None:
    relatorio = saude(
        [
            {
                "run_id": "run-b",
                "stage": "deteccao",
                "status": "success",
                "metricas": [
                    medida("drift.dataset_drift", "drift", "atencao", True, "booleano"),
                    medida("drift.variaveis_com_drift", "drift", "atencao", 13, "variaveis"),
                    medida("drift.proporcao_drift", "drift", "critico", 0.59, "proporcao"),
                ],
            }
        ],
        agregados=[medida("drift.alertas_total", "drift", "ok", 1, "alertas")],
    )
    drift = al.consolidar_drift(relatorio)

    assert drift["dataset_drift"] is True
    assert drift["variaveis_com_drift"] == 13
    assert drift["proporcao_drift"] == 0.59
    assert drift["alertas_total"] == 1
    assert drift["execucoes"] == ["run-b"]


def test_consolidar_drift_sem_dados() -> None:
    drift = al.consolidar_drift(saude([]))
    assert drift["dataset_drift"] is None
    assert drift["variaveis_com_drift"] is None
    assert drift["alertas_total"] is None
    assert drift["execucoes"] == []
