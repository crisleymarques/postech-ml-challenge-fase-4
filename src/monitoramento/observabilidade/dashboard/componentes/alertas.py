"""Apresentacao da aba de alertas."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..config import SEVERIDADE_CRITICAL, SEVERIDADE_WARNING
from ..indicadores import resumo_alertas
from .comuns import _tabela


def render_alertas(painel: dict[str, Any]) -> None:
    """Aba de alertas: lista por severidade, categoria e origem."""
    alertas = painel["alertas"]
    if not resumo_alertas(alertas):
        st.info("Relatório de alertas (F3-03) indisponível.")
        return

    resumo = alertas.get("resumo") or {}
    por_severidade = resumo.get("por_severidade") or {}
    colunas = st.columns(3)
    colunas[0].metric("Total", resumo.get("total", 0))
    colunas[1].metric("Críticos", por_severidade.get(SEVERIDADE_CRITICAL, 0))
    colunas[2].metric("Atenção", por_severidade.get(SEVERIDADE_WARNING, 0))

    lista = [alerta for alerta in alertas.get("alertas", []) or [] if isinstance(alerta, dict)]
    if not lista:
        st.info("Nenhum alerta registrado.")
        return

    severidades = sorted({alerta.get("severidade") for alerta in lista if alerta.get("severidade")})
    categorias = sorted({alerta.get("categoria") for alerta in lista if alerta.get("categoria")})
    colunas = st.columns(2)
    filtro_sev = colunas[0].multiselect("Severidade", severidades, default=severidades)
    filtro_cat = colunas[1].multiselect("Categoria", categorias, default=categorias)

    selecionados = [
        alerta
        for alerta in lista
        if alerta.get("severidade") in (filtro_sev or severidades)
        and alerta.get("categoria") in (filtro_cat or categorias)
    ]
    st.dataframe(_tabela(selecionados), use_container_width=True, hide_index=True)
