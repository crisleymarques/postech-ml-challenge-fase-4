"""Apresentacao da aba de drift."""

from __future__ import annotations

from typing import Any

import streamlit as st

from .comuns import _tabela


def render_drift(painel: dict[str, Any]) -> None:
    """Aba de drift: bloco consolidado e referencias aos relatorios."""
    alertas = painel["alertas"]
    if alertas is None or not isinstance(alertas.get("drift"), dict):
        st.info("Resultados de drift indisponíveis (artefato de alertas F3-03 ausente).")
        return

    drift = alertas["drift"]
    destaque = drift.get("execucao_resumo")
    if destaque:
        st.caption(
            f"Destaque histórico: execução {destaque}. Maior proporção de variáveis com drift, "
            "com desempate por quantidade e execução. Alertas: total histórico."
        )
    colunas = st.columns(4)
    presente = drift.get("dataset_drift")
    colunas[0].metric("Data drift", "—" if presente is None else ("sim" if presente else "não"))
    variaveis = drift.get("variaveis_com_drift")
    colunas[1].metric("Variáveis com drift", "—" if variaveis is None else int(variaveis))
    proporcao = drift.get("proporcao_drift")
    colunas[2].metric("Proporção", "—" if proporcao is None else f"{proporcao:.3f}")
    alertas_total = drift.get("alertas_total")
    colunas[3].metric("Alertas de drift", "—" if alertas_total is None else int(alertas_total))

    por_execucao = drift.get("por_execucao") or []
    if por_execucao:
        st.subheader("Drift por execução")
        st.dataframe(
            _tabela(por_execucao).rename(
                columns={
                    "run_id": "Execução",
                    "stage": "Etapa",
                    "dataset_drift": "Drift no conjunto",
                    "variaveis_com_drift": "Variáveis com drift",
                    "proporcao_drift": "Proporção",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    execucoes = drift.get("execucoes") or []
    if execucoes:
        st.caption("Execuções com medidas de drift: " + ", ".join(str(item) for item in execucoes))

    st.caption(
        "Relatórios detalhados: `reports/runs/<run_id>/drift.json`, `analise.json`, "
        "`drift.html` e `performance.json`."
    )
