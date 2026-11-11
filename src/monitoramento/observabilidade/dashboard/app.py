"""Aplicacao Streamlit do painel de monitoramento."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from monitoramento.observabilidade.dashboard.componentes.alertas import render_alertas
from monitoramento.observabilidade.dashboard.componentes.drift import render_drift
from monitoramento.observabilidade.dashboard.componentes.logs import render_logs
from monitoramento.observabilidade.dashboard.componentes.saude import render_saude
from monitoramento.observabilidade.dashboard.componentes.visao_geral import render_visao_geral
from monitoramento.observabilidade.dashboard.config import (
    ALERTAS_PADRAO,
    LOGS_RESUMO_PADRAO,
    RUNS_DIR_PADRAO,
    SAUDE_PADRAO,
)
from monitoramento.observabilidade.dashboard.servico import carregar_painel


def main() -> None:
    st.set_page_config(page_title="Monitoramento — Fase 4", layout="wide")
    st.title("Monitoramento do pipeline de Credit Scoring")
    st.caption(
        "Saúde do pipeline, infraestrutura, drift e alertas a partir dos artefatos persistidos."
    )

    st.sidebar.header("Fontes de dados")
    runs_dir = Path(st.sidebar.text_input("Execuções", RUNS_DIR_PADRAO.as_posix()))
    logs_path = Path(st.sidebar.text_input("Logs (F3-01)", LOGS_RESUMO_PADRAO.as_posix()))
    saude_path = Path(st.sidebar.text_input("Saúde (F3-02)", SAUDE_PADRAO.as_posix()))
    alertas_path = Path(st.sidebar.text_input("Alertas (F3-03)", ALERTAS_PADRAO.as_posix()))

    painel = carregar_painel(runs_dir, logs_path, saude_path, alertas_path)

    if not painel["execucoes"] and not any(painel["disponibilidade"].values()):
        st.info("Nenhuma execução ou artefato encontrado nos caminhos informados.")
        return

    abas = st.tabs(["Visão geral", "Saúde operacional", "Drift", "Alertas", "Logs e falhas"])
    with abas[0]:
        render_visao_geral(painel)
    with abas[1]:
        render_saude(painel)
    with abas[2]:
        render_drift(painel)
    with abas[3]:
        render_alertas(painel)
    with abas[4]:
        render_logs(painel)


if __name__ == "__main__":
    main()
