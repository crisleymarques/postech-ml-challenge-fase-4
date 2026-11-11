"""Apresentacao da aba de saude."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..config import CATEGORIA_AGREGADO, CATEGORIA_EXECUCAO, SITUACAO_ATENCAO, SITUACAO_CRITICO
from ..indicadores import contar_situacoes, medidas_de_saude
from .comuns import _tabela


def render_saude(painel: dict[str, Any]) -> None:
    """Aba de saude operacional: medidas por execucao e agregados."""
    saude = painel["saude"]
    if saude is None:
        st.info("Relatório de saúde (F3-02) indisponível.")
        return

    situacoes = contar_situacoes(saude)
    if situacoes:
        colunas = st.columns(3)
        colunas[0].metric("Medidas ok", situacoes.get("ok", 0))
        colunas[1].metric("Em atenção", situacoes.get(SITUACAO_ATENCAO, 0))
        colunas[2].metric("Críticas", situacoes.get(SITUACAO_CRITICO, 0))

    medidas = medidas_de_saude(saude)
    if not medidas:
        st.info("Nenhuma medida de saúde coletada.")
        return

    categorias = sorted({medida["categoria"] for medida in medidas if medida["categoria"]})
    selecionadas = st.multiselect("Filtrar por categoria", categorias, default=categorias)
    selecionadas_ou_todas = selecionadas or categorias

    st.subheader("Medidas por execução")
    execucao = [
        medida
        for medida in medidas
        if medida["escopo"] == CATEGORIA_EXECUCAO and medida["categoria"] in selecionadas_ou_todas
    ]
    st.dataframe(_tabela(execucao), use_container_width=True, hide_index=True)

    st.subheader("Agregados")
    agregados = [
        medida
        for medida in medidas
        if medida["escopo"] == CATEGORIA_AGREGADO and medida["categoria"] in selecionadas_ou_todas
    ]
    st.dataframe(_tabela(agregados), use_container_width=True, hide_index=True)
