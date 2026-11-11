"""Apresentacao da aba de logs."""

from __future__ import annotations

from typing import Any

import streamlit as st

from .comuns import _tabela


def render_logs(painel: dict[str, Any]) -> None:
    """Aba de logs: contagens por categoria/resultado e falhas."""
    logs = painel["logs"]
    if logs is None:
        st.info("Resumo dos logs (F3-01) indisponível.")
        return

    execucoes = logs.get("execucoes") or {}
    eventos = logs.get("eventos") or {}
    colunas = st.columns(3)
    colunas[0].metric("Execuções", execucoes.get("total", 0))
    colunas[1].metric("Eventos", eventos.get("total", 0))
    colunas[2].metric("Falhas", len(logs.get("falhas", []) or []))

    por_categoria = logs.get("por_categoria") or {}
    if por_categoria:
        st.subheader("Eventos por categoria")
        linhas = [
            {"categoria": categoria, **contagens}
            for categoria, contagens in sorted(por_categoria.items())
            if isinstance(contagens, dict)
        ]
        st.dataframe(_tabela(linhas), use_container_width=True, hide_index=True)

    falhas = [falha for falha in logs.get("falhas", []) or [] if isinstance(falha, dict)]
    if falhas:
        st.subheader("Falhas registradas")
        st.dataframe(_tabela(falhas), use_container_width=True, hide_index=True)
    else:
        st.success("Nenhuma falha de execução ou de contrato registrada.")
