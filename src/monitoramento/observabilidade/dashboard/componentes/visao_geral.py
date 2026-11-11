"""Apresentacao da aba de visao geral."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ..indicadores import indicadores
from .comuns import _origem_artefato, _tabela


def render_visao_geral(painel: dict[str, Any]) -> None:
    """Aba de visao geral: indicadores, saude dos artefatos e execucoes."""
    ind = indicadores(painel)
    colunas = st.columns(5)
    colunas[0].metric("Execuções", ind["execucoes"])
    taxa = ind["taxa_sucesso"]
    colunas[1].metric("Taxa de sucesso", "—" if taxa is None else f"{taxa:.1%}")
    colunas[2].metric("Falhas", ind["falhas"])
    colunas[3].metric("Rejeições", ind["rejeicoes"])
    colunas[4].metric("Alertas", "—" if ind["alertas_total"] is None else ind["alertas_total"])

    st.subheader("Alertas e drift")
    colunas = st.columns(4)
    colunas[0].metric(
        "Alertas críticos", "—" if ind["alertas_critical"] is None else ind["alertas_critical"]
    )
    colunas[1].metric(
        "Alertas de atenção", "—" if ind["alertas_warning"] is None else ind["alertas_warning"]
    )
    drift = ind["dataset_drift"]
    colunas[2].metric("Data drift", "—" if drift is None else ("sim" if drift else "não"))
    variaveis = ind["variaveis_com_drift"]
    colunas[3].metric("Variáveis com drift", "—" if variaveis is None else int(variaveis))

    st.subheader("Saúde dos artefatos")
    disponibilidade = painel["disponibilidade"]
    st.dataframe(
        _tabela(
            [
                {
                    "artefato": "logs (F3-01)",
                    "situação": _origem_artefato(painel["logs"], disponibilidade["logs"]),
                },
                {
                    "artefato": "saúde (F3-02)",
                    "situação": _origem_artefato(painel["saude"], disponibilidade["saude"]),
                },
                {
                    "artefato": "alertas (F3-03)",
                    "situação": _origem_artefato(painel["alertas"], disponibilidade["alertas"]),
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Execuções por etapa")
    por_etapa = painel["resumo_execucoes"]["por_etapa"]
    if por_etapa:
        st.bar_chart(pd.DataFrame({"execuções": por_etapa}))
    else:
        st.info("Nenhuma execução registrada.")

    incompletas = painel["resumo_execucoes"]["incompletas"]
    if incompletas:
        st.warning("Execuções sem evento terminal (incompletas): " + ", ".join(incompletas))

    execucoes = painel["execucoes"]
    if execucoes:
        st.subheader("Histórico de execuções")
        st.dataframe(_tabela(execucoes), use_container_width=True, hide_index=True)
