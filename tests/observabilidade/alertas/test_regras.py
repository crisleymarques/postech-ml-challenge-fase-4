"""Testes de regras dos alertas."""

from __future__ import annotations

from monitoramento.observabilidade.alertas import regras as al

from .apoio import medida, resumo_logs, saude


def test_alertas_de_medidas_por_situacao() -> None:
    relatorio = saude(
        [
            {
                "run_id": "run-a",
                "stage": "validacao",
                "status": "rejected",
                "metricas": [
                    medida("pipeline.status_execucao", "pipeline", "atencao", "rejected"),
                    medida("qualidade.violacoes_total", "qualidade", "ok", 0),
                ],
            },
            {
                "run_id": "run-b",
                "stage": "deteccao",
                "status": "success",
                "metricas": [
                    medida("drift.proporcao_drift", "drift", "critico", 0.59, "proporcao")
                ],
            },
        ]
    )
    mapa = {"atencao": "warning", "critico": "critical"}
    alertas = al.alertas_de_medidas(relatorio, mapa)

    assert len(alertas) == 2
    assert alertas[0]["severidade"] == "warning"
    assert alertas[0]["origem"] == "run-a"
    assert alertas[0]["fonte"] == "saude"
    assert alertas[1]["severidade"] == "critical"
    assert alertas[1]["categoria"] == "drift"


def test_alertas_de_medidas_ignora_ok() -> None:
    relatorio = saude(
        [
            {
                "run_id": "run-a",
                "stage": "validacao",
                "status": "success",
                "metricas": [medida("pipeline.duracao_s", "pipeline", "ok", 1.0, "segundos")],
            }
        ]
    )
    assert al.alertas_de_medidas(relatorio, {"atencao": "warning", "critico": "critical"}) == []


def test_alertas_de_falhas() -> None:
    resumo = resumo_logs(
        falhas=[
            {
                "run_id": "run-f104-validacao",
                "stage": "validacao",
                "categoria": "validacao",
                "event": "rejected",
                "level": "WARNING",
                "error_code": "QUALIDADE_REPROVADA",
                "timestamp": "2026-11-09T00:00:00.000000Z",
            }
        ]
    )
    alertas = al.alertas_de_falhas(resumo, "critical")

    assert len(alertas) == 1
    alerta = alertas[0]
    assert alerta["severidade"] == "critical"
    assert alerta["error_code"] == "QUALIDADE_REPROVADA"
    assert alerta["fonte"] == "logs"
    assert "QUALIDADE_REPROVADA" in alerta["descricao"]


def test_alertas_de_drift() -> None:
    resumo = resumo_logs(
        alertas_drift=[
            {
                "run_id": "run-f203-deteccao",
                "stage": "deteccao",
                "scenario_id": "cenario-01",
                "batch_id": "lote-01",
                "timestamp": "2026-11-09T00:00:00.000000Z",
            }
        ]
    )
    alertas = al.alertas_de_drift(resumo, "warning")

    assert len(alertas) == 1
    assert alertas[0]["severidade"] == "warning"
    assert alertas[0]["categoria"] == "drift"
    assert alertas[0]["evidencia"]["scenario_id"] == "cenario-01"


def test_ids_de_alerta_sao_unicos() -> None:
    relatorio = saude(
        [
            {
                "run_id": "run-a",
                "stage": "validacao",
                "status": "rejected",
                "metricas": [medida("qualidade.violacoes_total", "qualidade", "atencao", 7)],
            }
        ]
    )
    resumo = resumo_logs(
        falhas=[
            {
                "run_id": "run-a",
                "stage": "validacao",
                "event": "rejected",
                "error_code": "QUALIDADE_REPROVADA",
            }
        ],
        alertas_drift=[{"run_id": "run-b", "stage": "deteccao"}],
    )
    alertas = (
        al.alertas_de_medidas(relatorio, {"atencao": "warning", "critico": "critical"})
        + al.alertas_de_falhas(resumo, "critical")
        + al.alertas_de_drift(resumo, "warning")
    )
    ids = [alerta["id"] for alerta in alertas]
    assert len(ids) == len(set(ids))
