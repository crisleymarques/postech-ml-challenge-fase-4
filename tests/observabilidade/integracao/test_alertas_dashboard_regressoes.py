"""Verifica a apresentacao dos resultados F3-03 propagados ao dashboard."""

from streamlit.testing.v1 import AppTest

from monitoramento.observabilidade.alertas.drift import consolidar_drift
from tests.observabilidade.alertas.test_alertas_regressoes import execucao


def pagina_drift() -> None:
    import streamlit as st

    from monitoramento.observabilidade.dashboard.componentes.drift import render_drift

    render_drift({"alertas": {"drift": st.session_state["drift"]}})


def test_dashboard_exibe_mesma_execucao_no_destaque_e_preserva_controle() -> None:
    drift = consolidar_drift(
        {
            "execucoes": [
                execucao("a-drift", True, 13, 0.59),
                execucao("z-controle", False, 0, 0.0),
            ],
            "agregados": [{"id": "drift.alertas_total", "valor": 3}],
        }
    )
    app = AppTest.from_function(pagina_drift)
    app.session_state["drift"] = drift
    app.run()

    assert not app.exception
    assert [m.value for m in app.metric] == ["sim", "13", "0.590", "3"]
    assert "Destaque histórico: execução a-drift" in app.caption[0].value
    tabela = app.dataframe[0].value.set_index("Execução")
    assert bool(tabela.loc["a-drift", "Drift no conjunto"]) is True
    assert bool(tabela.loc["z-controle", "Drift no conjunto"]) is False
    assert tabela.loc["z-controle", "Proporção"] == 0


def test_dashboard_nao_inventa_dataset_drift_na_analise_estatistica() -> None:
    analise = execucao("analise", None, 2, 1.0)
    analise["stage"] = "analise"
    analise["metricas"] = analise["metricas"][1:]
    drift = consolidar_drift(
        {
            "execucoes": [
                execucao("deteccao", True, 13, 0.59),
                analise,
            ]
        }
    )
    app = AppTest.from_function(pagina_drift)
    app.session_state["drift"] = drift
    app.run()

    assert not app.exception
    assert [m.value for m in app.metric] == ["—", "2", "1.000", "—"]
    assert "Destaque histórico: execução analise" in app.caption[0].value
    assert len(app.dataframe[0].value) == 2
