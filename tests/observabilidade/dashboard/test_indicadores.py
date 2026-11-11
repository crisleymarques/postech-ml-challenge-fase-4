"""Testes de indicadores do painel."""

from __future__ import annotations

from pathlib import Path

from monitoramento.observabilidade.dashboard import indicadores


def test_resumir_execucoes(tmp_path: Path) -> None:
    execucoes = [
        {"run_id": "a", "stage": "validacao", "status": "success", "incompleta": False},
        {"run_id": "b", "stage": "validacao", "status": "rejected", "incompleta": False},
        {"run_id": "c", "stage": "deteccao", "status": "incomplete", "incompleta": True},
    ]
    resumo = indicadores.resumir_execucoes(execucoes)

    assert resumo["total"] == 3
    assert resumo["por_status"] == {"success": 1, "rejected": 1, "incomplete": 1}
    assert resumo["por_etapa"] == {"validacao": 2, "deteccao": 1}
    assert resumo["incompletas"] == ["c"]


def test_taxa_sucesso() -> None:
    execucoes = [
        {"status": "success"},
        {"status": "success"},
        {"status": "rejected"},
        {"status": "failed"},
    ]
    assert indicadores.taxa_sucesso(execucoes) == 0.5
    assert indicadores.taxa_sucesso([]) is None


def test_indicadores() -> None:
    painel = {
        "execucoes": [{"status": "success"}, {"status": "failed"}],
        "resumo_execucoes": {
            "total": 2,
            "por_status": {"success": 1, "failed": 1},
            "por_etapa": {"treinamento": 1, "deteccao": 1},
            "incompletas": [],
        },
        "alertas": {
            "resumo": {"total": 3, "por_severidade": {"warning": 2, "critical": 1}},
            "drift": {"dataset_drift": True, "variaveis_com_drift": 13},
        },
    }
    ind = indicadores.indicadores(painel)

    assert ind["execucoes"] == 2
    assert ind["taxa_sucesso"] == 0.5
    assert ind["falhas"] == 1
    assert ind["alertas_total"] == 3
    assert ind["alertas_critical"] == 1
    assert ind["dataset_drift"] is True
    assert ind["variaveis_com_drift"] == 13


def test_indicadores_sem_alertas() -> None:
    painel = {
        "execucoes": [],
        "resumo_execucoes": {
            "total": 0,
            "por_status": {},
            "por_etapa": {},
            "incompletas": [],
        },
        "alertas": None,
    }
    ind = indicadores.indicadores(painel)

    assert ind["execucoes"] == 0
    assert ind["taxa_sucesso"] is None
    assert ind["alertas_total"] is None
    assert ind["alertas_warning"] is None
    assert ind["alertas_critical"] is None
    assert ind["dataset_drift"] is None


def test_contar_situacoes() -> None:
    saude = {"resumo": {"por_situacao": {"ok": 10, "atencao": 2, "critico": 1}}}
    assert indicadores.contar_situacoes(saude) == {"ok": 10, "atencao": 2, "critico": 1}
    assert indicadores.contar_situacoes(None) == {}


def test_medidas_de_saude() -> None:
    saude = {
        "execucoes": [
            {
                "run_id": "run-a",
                "metricas": [
                    {
                        "id": "pipeline.duracao_s",
                        "categoria": "pipeline",
                        "valor": 1.5,
                        "unidade": "segundos",
                        "situacao": "ok",
                    }
                ],
            }
        ],
        "agregados": [
            {
                "id": "drift.alertas_total",
                "categoria": "drift",
                "valor": 3,
                "unidade": "alertas",
                "situacao": "ok",
            }
        ],
    }
    medidas = indicadores.medidas_de_saude(saude)

    assert len(medidas) == 2
    assert medidas[0]["origem"] == "run-a"
    assert medidas[0]["escopo"] == "execucao"
    assert medidas[1]["origem"] == "agregado"
    assert medidas[1]["escopo"] == "agregado"


def test_medidas_de_saude_ausente() -> None:
    assert indicadores.medidas_de_saude(None) == []
