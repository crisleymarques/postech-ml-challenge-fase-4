"""Classificacao operacional de etapas, eventos e resultados."""

from __future__ import annotations

from typing import Any

CATEGORIA_POR_ETAPA = {
    "preparacao": "ingestao",
    "validacao": "validacao",
    "treinamento": "treinamento",
    "simulacao": "simulacao",
    "predicao": "predicao",
    "deteccao": "drift",
    "analise": "drift",
    "avaliacao": "avaliacao",
}
CATEGORIAS_POR_ETAPA = {"preparacao": ("ingestao", "transformacao")}
CATEGORIAS_CONHECIDAS = (
    "ingestao",
    "transformacao",
    "validacao",
    "treinamento",
    "simulacao",
    "predicao",
    "drift",
    "avaliacao",
)
CATEGORIA_DESCONHECIDA = "desconhecida"
RESULTADO_POR_EVENTO = {
    "started": "inicio",
    "completed": "sucesso",
    "rejected": "rejeicao",
    "failed": "falha",
    "skipped": "ignorado",
    "drift_alert": "alerta",
}
RESULTADO_DESCONHECIDO = "desconhecido"
ORDEM_RESULTADOS = ("inicio", "sucesso", "rejeicao", "falha", "alerta", "ignorado", "desconhecido")
STATUS_TERMINAL = {"completed": "success", "rejected": "rejected", "failed": "failed"}
STATUS_INCOMPLETO = "incomplete"
RESULTADO_POR_STATUS = {
    "success": "sucesso",
    "rejected": "rejeicao",
    "failed": "falha",
    "incomplete": "incompleto",
}


def categoria_da_etapa(stage: str | None) -> str:
    """Categoria operacional principal de uma etapa."""
    if stage is None:
        return CATEGORIA_DESCONHECIDA
    return CATEGORIA_POR_ETAPA.get(stage, CATEGORIA_DESCONHECIDA)


def categorias_da_etapa(stage: str | None) -> tuple[str, ...]:
    """Categorias operacionais que uma etapa alimenta."""
    if stage is None:
        return (CATEGORIA_DESCONHECIDA,)
    return CATEGORIAS_POR_ETAPA.get(stage, (categoria_da_etapa(stage),))


def resultado_do_evento(evento: str | None) -> str:
    """Resultado operacional derivado do tipo de evento."""
    if evento is None:
        return RESULTADO_DESCONHECIDO
    return RESULTADO_POR_EVENTO.get(evento, RESULTADO_DESCONHECIDO)


def resultado_do_status(status: str | None) -> str:
    """Resultado operacional derivado do status de uma execucao."""
    if status is None:
        return RESULTADO_POR_STATUS[STATUS_INCOMPLETO]
    return RESULTADO_POR_STATUS.get(status, status)


def inferir_status(eventos: list[dict[str, Any]]) -> str:
    """Infere o status da execucao pelo ultimo evento terminal observado."""
    for evento in reversed(eventos):
        status = STATUS_TERMINAL.get(evento.get("event"))
        if status is not None:
            return status
    return STATUS_INCOMPLETO


def classificar_evento(evento: dict[str, Any]) -> dict[str, Any]:
    """Acrescenta `categoria` e `resultado` ao evento, sem remove-lo do contrato."""
    classificado = dict(evento)
    classificado["categoria"] = categoria_da_etapa(evento.get("stage"))
    classificado["resultado"] = resultado_do_evento(evento.get("event"))
    return classificado
