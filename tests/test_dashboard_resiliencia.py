"""Regressoes de eventos corrompidos e alertas indisponiveis na interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from monitoramento import dashboard as dash
from tests.test_dashboard import escrever_execucao, evento


def pagina() -> None:
    from monitoramento.dashboard import main

    main()


@pytest.mark.parametrize("invalido", [None, [], 1, "texto", True])
def test_evento_nao_objeto_nao_impede_outras_execucoes(tmp_path, invalido):
    runs = tmp_path / "runs"
    ruim = escrever_execucao(runs, "ruim")
    (ruim / "events.jsonl").write_text(json.dumps(invalido) + "\n", encoding="utf-8")
    escrever_execucao(
        runs,
        "boa",
        manifesto={
            "contract_version": "1.0",
            "run_id": "boa",
            "stage": "predicao",
            "status": "success",
        },
        eventos=[evento("boa", "predicao", "completed")],
    )
    painel = dash.carregar_painel(runs, tmp_path / "logs", tmp_path / "saude", tmp_path / "alertas")
    assert painel["resumo_execucoes"]["por_status"] == {"success": 1, "incomplete": 1}
    assert painel["execucoes"][1]["event_count"] == 0


def test_eventos_invalidos_entre_validos_preservam_eventos_validos(tmp_path):
    caminho = tmp_path / "events.jsonl"
    caminho.write_text(
        'null\n{"event":"started"}\n[]\nnao-json\n{"event":"completed"}\n',
        encoding="utf-8",
    )
    assert [e["event"] for e in dash.ler_eventos(caminho)] == ["started", "completed"]


@pytest.mark.parametrize(
    "erro", [PermissionError("sem acesso"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalido")]
)
def test_falha_de_leitura_de_eventos_isolada(tmp_path, monkeypatch, erro):
    runs = tmp_path / "runs"
    ruim = escrever_execucao(runs, "ruim", eventos=[evento("ruim", "predicao", "started")])
    escrever_execucao(runs, "outra", manifesto={"status": "failed", "stage": "predicao"})
    original = Path.open

    def abrir(caminho, *args, **kwargs):
        if caminho == ruim / "events.jsonl":
            raise erro
        return original(caminho, *args, **kwargs)

    monkeypatch.setattr(Path, "open", abrir)
    execucoes = dash.carregar_execucoes(runs)
    assert [e["status"] for e in execucoes] == ["failed", "incomplete"]


@pytest.mark.parametrize("conteudo", [None, "{", "[]", "{}", '{"resumo": []}'])
def test_interface_distingue_alertas_indisponiveis_de_zero(tmp_path, monkeypatch, conteudo):
    runs = tmp_path / "runs"
    ruim = escrever_execucao(runs, "ruim")
    (ruim / "events.jsonl").write_text("null\n[]\n", encoding="utf-8")
    escrever_execucao(runs, "falha", manifesto={"status": "failed", "stage": "predicao"})
    destino = tmp_path / "alertas.json"
    if conteudo is not None:
        destino.write_text(conteudo, encoding="utf-8")
    monkeypatch.setattr(dash, "RUNS_DIR_PADRAO", runs)
    monkeypatch.setattr(dash, "LOGS_RESUMO_PADRAO", tmp_path / "logs.json")
    monkeypatch.setattr(dash, "SAUDE_PADRAO", tmp_path / "saude.json")
    monkeypatch.setattr(dash, "ALERTAS_PADRAO", destino)

    painel = dash.carregar_painel(runs, tmp_path / "logs", tmp_path / "saude", destino)
    assert painel["disponibilidade"]["alertas"] is False
    ind = dash.indicadores(painel)
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
    monkeypatch.setattr(dash, "RUNS_DIR_PADRAO", tmp_path / "runs")
    monkeypatch.setattr(dash, "LOGS_RESUMO_PADRAO", tmp_path / "logs")
    monkeypatch.setattr(dash, "SAUDE_PADRAO", tmp_path / "saude")
    monkeypatch.setattr(dash, "ALERTAS_PADRAO", destino)
    app = AppTest.from_function(pagina).run(timeout=20)
    assert not app.exception
    metricas = {m.label: m.value for m in app.metric}
    assert all(metricas[k] == "0" for k in ("Alertas", "Alertas críticos", "Alertas de atenção"))
    assert any(i.value == "Nenhum alerta registrado." for i in app.info)


def test_interface_integra_saude_infraestrutura_drift_e_alertas(tmp_path, monkeypatch):
    from monitoramento import alertas, logs, metricas_saude
    from tests.test_saude_alertas_regressoes import RAIZ, gerar_fontes

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
    monkeypatch.setattr(dash, "RUNS_DIR_PADRAO", tmp_path / "runs")
    monkeypatch.setattr(dash, "LOGS_RESUMO_PADRAO", tmp_path / "logs/resumo.json")
    monkeypatch.setattr(dash, "SAUDE_PADRAO", tmp_path / "saude/saude.json")
    monkeypatch.setattr(dash, "ALERTAS_PADRAO", tmp_path / "alertas/alertas.json")
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
