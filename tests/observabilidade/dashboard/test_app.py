"""Testes de app do painel."""

from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from monitoramento.observabilidade.dashboard import app as painel_app
from monitoramento.observabilidade.dashboard import indicadores, servico
from tests.observabilidade.dashboard.apoio import escrever_execucao, evento


def pagina() -> None:
    from monitoramento.observabilidade.dashboard.app import main

    main()


@pytest.mark.parametrize("conteudo", [None, "{", "[]", "{}", '{"resumo": []}'])
def test_interface_distingue_alertas_indisponiveis_de_zero(tmp_path, monkeypatch, conteudo):
    runs = tmp_path / "runs"
    ruim = escrever_execucao(runs, "ruim")
    (ruim / "events.jsonl").write_text("null\n[]\n", encoding="utf-8")
    escrever_execucao(runs, "falha", manifesto={"status": "failed", "stage": "predicao"})
    destino = tmp_path / "alertas.json"
    if conteudo is not None:
        destino.write_text(conteudo, encoding="utf-8")
    monkeypatch.setattr(painel_app, "RUNS_DIR_PADRAO", runs)
    monkeypatch.setattr(painel_app, "LOGS_RESUMO_PADRAO", tmp_path / "logs.json")
    monkeypatch.setattr(painel_app, "SAUDE_PADRAO", tmp_path / "saude.json")
    monkeypatch.setattr(painel_app, "ALERTAS_PADRAO", destino)

    painel = servico.carregar_painel(runs, tmp_path / "logs", tmp_path / "saude", destino)
    assert painel["disponibilidade"]["alertas"] is False
    ind = indicadores.indicadores(painel)
    assert all(ind[k] is None for k in ("alertas_total", "alertas_warning", "alertas_critical"))
    app = AppTest.from_function(pagina).run(timeout=20)
    assert not app.exception
    metricas = {m.label: m.value for m in app.metric}
    assert metricas["Falhas"] == "1"
    assert all(metricas[k] == "—" for k in ("Alertas", "Alertas críticos", "Alertas de atenção"))
    assert any("alertas (F3-03) indisponível" in i.value for i in app.info)
    assert len(app.tabs) == 5


def test_interface_preserva_zero_confirmado(tmp_path, monkeypatch):
    destino = tmp_path / "alertas.json"
    destino.write_text(
        json.dumps(
            {
                "resumo": {"total": 0, "por_severidade": {"warning": 0, "critical": 0}},
                "alertas": [],
                "drift": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(painel_app, "RUNS_DIR_PADRAO", tmp_path / "runs")
    monkeypatch.setattr(painel_app, "LOGS_RESUMO_PADRAO", tmp_path / "logs")
    monkeypatch.setattr(painel_app, "SAUDE_PADRAO", tmp_path / "saude")
    monkeypatch.setattr(painel_app, "ALERTAS_PADRAO", destino)
    app = AppTest.from_function(pagina).run(timeout=20)
    assert not app.exception
    metricas = {m.label: m.value for m in app.metric}
    assert all(metricas[k] == "0" for k in ("Alertas", "Alertas críticos", "Alertas de atenção"))
    assert any(i.value == "Nenhum alerta registrado." for i in app.info)


def test_interface_integra_saude_infraestrutura_drift_e_alertas(tmp_path, monkeypatch):
    from monitoramento.observabilidade import alertas, logs, metricas_saude
    from tests.observabilidade.integracao.test_saude_alertas_regressoes import RAIZ, gerar_fontes

    gerar_fontes(tmp_path, 0.49)
    deteccao = escrever_execucao(
        tmp_path / "runs",
        "deteccao",
        manifesto={
            "contract_version": "1.0",
            "run_id": "deteccao",
            "stage": "deteccao",
            "status": "success",
            "drift": {"dataset_drift": True},
            "started_at": "2026-11-09T00:00:00Z",
            "finished_at": "2026-11-09T00:00:02Z",
            "inputs": [{"row_count": 100}],
        },
        eventos=[
            evento("deteccao", "deteccao", "drift_alert"),
            evento("deteccao", "deteccao", "completed"),
        ],
    )
    (deteccao / "drift.json").write_text(
        json.dumps(
            {
                "summary": {"dataset_drift": True, "drifted_columns": 2, "drift_share": 0.5},
            }
        ),
        encoding="utf-8",
    )
    logs.gerar(tmp_path / "runs", tmp_path / "logs")
    metricas_saude.gerar(
        tmp_path / "runs", tmp_path / "saude", RAIZ / "configs/metricas_saude.json"
    )
    integrado = alertas.gerar(
        tmp_path / "saude/saude.json",
        tmp_path / "logs/resumo.json",
        RAIZ / "configs/alertas.json",
        tmp_path / "alertas",
    )
    monkeypatch.setattr(painel_app, "RUNS_DIR_PADRAO", tmp_path / "runs")
    monkeypatch.setattr(painel_app, "LOGS_RESUMO_PADRAO", tmp_path / "logs/resumo.json")
    monkeypatch.setattr(painel_app, "SAUDE_PADRAO", tmp_path / "saude/saude.json")
    monkeypatch.setattr(painel_app, "ALERTAS_PADRAO", tmp_path / "alertas/alertas.json")
    app = AppTest.from_function(pagina).run(timeout=20)
    assert not app.exception
    assert len(app.tabs) == 5
    metricas = {m.label: m.value for m in app.metric}
    assert metricas["Execuções"] == "2"
    assert metricas["Taxa de sucesso"] == "100.0%"
    assert metricas["Data drift"] == "sim"
    assert metricas["Alertas"] == str(integrado["resumo"]["total"])
    assert int(metricas["Alertas críticos"]) >= 1
    saude = app.tabs[1].dataframe[0].value.set_index("id")
    assert float(saude.loc["modelo.roc_auc", "valor"]) == 0.49
    infra = app.tabs[1].dataframe[1].value
    assert {"infra.espaco_livre_bytes", "infra.escrita_ok"} <= set(infra["id"])
    assert app.tabs[2].dataframe[0].value.iloc[0]["Execução"] == "deteccao"
    assert not app.tabs[3].dataframe[0].value.empty
    assert app.tabs[4].metric[1].value == "3"


def test_entrypoint_por_arquivo_abre_com_fontes_temporarias(tmp_path, monkeypatch):
    from pathlib import Path

    from monitoramento.observabilidade.dashboard import config

    escrever_execucao(
        tmp_path / "runs", "falha", manifesto={"status": "failed", "stage": "predicao"}
    )
    monkeypatch.setattr(config, "RUNS_DIR_PADRAO", tmp_path / "runs")
    monkeypatch.setattr(config, "LOGS_RESUMO_PADRAO", tmp_path / "logs.json")
    monkeypatch.setattr(config, "SAUDE_PADRAO", tmp_path / "saude.json")
    monkeypatch.setattr(config, "ALERTAS_PADRAO", tmp_path / "alertas.json")

    caminho = Path(painel_app.__file__)
    app = AppTest.from_file(str(caminho)).run(timeout=20)

    assert not app.exception
    assert len(app.tabs) == 5
    assert {m.label: m.value for m in app.metric}["Falhas"] == "1"
