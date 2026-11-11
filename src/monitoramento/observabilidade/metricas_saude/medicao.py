"""Composicao das medidas por execucao e agregadas."""

from __future__ import annotations

from typing import Any

from .catalogo import aplicavel
from .classificacao import DISPONIVEL, INDISPONIVEL, situacao_medida
from .coletores import coletar_metrica


def montar_medida(metrica: dict[str, Any], valor: Any) -> dict[str, Any]:
    """Descreve uma medida coletada conforme o catalogo."""
    return {
        "id": metrica["id"],
        "categoria": metrica["categoria"],
        "nome": metrica["nome"],
        "unidade": metrica["unidade"],
        "valor": valor,
        "disponibilidade": DISPONIVEL if valor is not None else INDISPONIVEL,
        "situacao": situacao_medida(metrica, valor),
    }


def medir_execucao(metrica: dict[str, Any], contexto: dict[str, Any]) -> dict[str, Any] | None:
    """Mede uma metrica de execucao, ignorando etapas nao aplicaveis."""
    if not aplicavel(metrica, contexto.get("stage")):
        return None
    return montar_medida(metrica, coletar_metrica(metrica, contexto))


def medir_agregado(metrica: dict[str, Any], contexto: dict[str, Any]) -> dict[str, Any]:
    """Mede uma metrica de escopo agregado."""
    return montar_medida(metrica, coletar_metrica(metrica, contexto))
