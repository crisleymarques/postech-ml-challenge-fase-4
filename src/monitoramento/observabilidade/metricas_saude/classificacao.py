"""Classificacao das medidas segundo os criterios do catalogo."""

from __future__ import annotations

from typing import Any, Callable

DIRECAO_MAIOR_PIOR = "maior_pior"

DIRECAO_MENOR_PIOR = "menor_pior"

SITUACAO_OK = "ok"

SITUACAO_ATENCAO = "atencao"

SITUACAO_CRITICO = "critico"

SITUACAO_INDISPONIVEL = "indisponivel"

SITUACAO_NAO_APLICAVEL = "nao_aplicavel"

ORDEM_SITUACOES = (
    SITUACAO_OK,
    SITUACAO_ATENCAO,
    SITUACAO_CRITICO,
    SITUACAO_INDISPONIVEL,
    SITUACAO_NAO_APLICAVEL,
)

DISPONIVEL = "available"

INDISPONIVEL = "not_available"

NAO_APLICAVEL = "not_applicable"

SITUACAO_POR_STATUS = {
    "success": SITUACAO_OK,
    "rejected": SITUACAO_ATENCAO,
    "failed": SITUACAO_CRITICO,
    "incomplete": SITUACAO_CRITICO,
}

SITUACAO_BOOLEANO_ALERTA = {False: SITUACAO_OK, True: SITUACAO_ATENCAO}

SITUACAO_BOOLEANO_FALHA = {False: SITUACAO_CRITICO, True: SITUACAO_OK}

AVALIADORES: dict[str, Callable[[Any], str]] = {
    "pipeline.status_execucao": lambda valor: SITUACAO_POR_STATUS.get(valor, SITUACAO_INDISPONIVEL),
    "drift.dataset_drift": lambda valor: SITUACAO_BOOLEANO_ALERTA.get(valor, SITUACAO_INDISPONIVEL),
    "drift.degradacao_conceito": lambda valor: SITUACAO_BOOLEANO_ALERTA.get(
        valor, SITUACAO_INDISPONIVEL
    ),
    "infra.escrita_ok": lambda valor: SITUACAO_BOOLEANO_FALHA.get(valor, SITUACAO_INDISPONIVEL),
}


def situacao_medida(metrica: dict[str, Any], valor: Any) -> str:
    """Situacao da medida segundo os limiares e o criterio do catalogo."""
    if valor is None:
        return SITUACAO_INDISPONIVEL

    avaliador = AVALIADORES.get(metrica["id"])
    if avaliador is not None:
        return avaliador(valor)

    direcao = metrica.get("direcao")
    if direcao == DIRECAO_MAIOR_PIOR:

        def atingiu(limiar: Any) -> bool:
            return valor >= limiar

    elif direcao == DIRECAO_MENOR_PIOR:

        def atingiu(limiar: Any) -> bool:
            return valor < limiar

    else:
        return SITUACAO_OK

    limiar_falha = metrica.get("limiar_falha")
    if limiar_falha is not None and atingiu(limiar_falha):
        return SITUACAO_CRITICO
    limiar_degracao = metrica.get("limiar_degracao")
    if limiar_degracao is not None and atingiu(limiar_degracao):
        return SITUACAO_ATENCAO
    return SITUACAO_OK
