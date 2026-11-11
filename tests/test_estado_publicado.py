"""Leitura de execucoes parcialmente publicadas pelos consumidores."""

from __future__ import annotations

import pytest

from monitoramento.infraestrutura.observabilidade import status_publicado


@pytest.mark.parametrize(
    "caso",
    [
        "valido",
        "sem_manifesto",
        "sem_terminal",
        "outro_run",
        "outra_etapa",
        "contrato_invalido",
        "falha_posterior",
        "alerta_ausente",
        "alerta_presente",
    ],
)
def test_sucesso_exige_publicacao_coerente(caso):
    manifesto = {
        "run_id": "run",
        "stage": "deteccao",
        "contract_version": "1.0",
        "status": "success",
    }
    eventos = [{"run_id": "run", "stage": "deteccao", "event": "completed"}]
    if caso == "sem_manifesto":
        manifesto = {}
    elif caso == "sem_terminal":
        eventos = []
    elif caso == "outro_run":
        eventos[0]["run_id"] = "outro"
    elif caso == "outra_etapa":
        eventos[0]["stage"] = "treinamento"
    elif caso == "contrato_invalido":
        manifesto["contract_version"] = "2.0"
    elif caso == "falha_posterior":
        eventos.append({**eventos[0], "event": "failed"})
    elif caso in ("alerta_ausente", "alerta_presente"):
        manifesto["drift"] = {"dataset_drift": True}
        if caso == "alerta_presente":
            eventos.insert(0, {**eventos[0], "event": "drift_alert"})
    esperado = "success" if caso in ("valido", "alerta_presente") else "incomplete"
    assert status_publicado(manifesto, eventos) == esperado


def test_falha_sem_manifesto_permanece_identificavel():
    assert status_publicado({}, [{"event": "failed"}]) == "failed"
