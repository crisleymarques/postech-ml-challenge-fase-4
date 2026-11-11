"""Agregacoes e indicadores puros do painel."""

from __future__ import annotations

from typing import Any

from .config import (
    CATEGORIA_AGREGADO,
    CATEGORIA_EXECUCAO,
    SEVERIDADE_CRITICAL,
    SEVERIDADE_WARNING,
    STATUS_FALHA,
    STATUS_REJEICAO,
    STATUS_SUCESSO,
)


def resumo_alertas(artefato: dict[str, Any] | None) -> dict[str, Any]:
    """Retorna contagens confirmadas; um resumo ausente ou invalido nao e zero."""
    resumo = (artefato or {}).get("resumo")
    if not isinstance(resumo, dict):
        return {}
    severidades = resumo.get("por_severidade")
    if not isinstance(severidades, dict):
        return {}
    contagens = [
        resumo.get("total"),
        *(severidades.get(s) for s in (SEVERIDADE_WARNING, SEVERIDADE_CRITICAL)),
    ]
    if any(type(valor) is not int or valor < 0 for valor in contagens):
        return {}
    return resumo


def resumir_execucoes(execucoes: list[dict[str, Any]]) -> dict[str, Any]:
    """Consolida as execucoes por status e por etapa."""
    por_status: dict[str, int] = {}
    por_etapa: dict[str, int] = {}
    for execucao in execucoes:
        status = execucao["status"]
        por_status[status] = por_status.get(status, 0) + 1
        etapa = execucao.get("stage") or "desconhecida"
        por_etapa[etapa] = por_etapa.get(etapa, 0) + 1
    return {
        "total": len(execucoes),
        "por_status": por_status,
        "por_etapa": dict(sorted(por_etapa.items())),
        "incompletas": [execucao["run_id"] for execucao in execucoes if execucao["incompleta"]],
    }


def taxa_sucesso(execucoes: list[dict[str, Any]]) -> float | None:
    """Proporcao de execucoes concluidas com status 'success'."""
    if not execucoes:
        return None
    sucessos = sum(1 for execucao in execucoes if execucao["status"] == STATUS_SUCESSO)
    return round(sucessos / len(execucoes), 6)


def contar_situacoes(saude: dict[str, Any] | None) -> dict[str, int]:
    """Contagens de situacao das medidas de saude, quando disponiveis."""
    resumo = (saude or {}).get("resumo") or {}
    por_situacao = resumo.get("por_situacao")
    return por_situacao if isinstance(por_situacao, dict) else {}


def indicadores(painel: dict[str, Any]) -> dict[str, Any]:
    """Indicadores sinteticos usados na visao geral do painel."""
    execucoes = painel["execucoes"]
    por_status = painel["resumo_execucoes"]["por_status"]
    alertas = painel.get("alertas") or {}
    resumo = resumo_alertas(alertas)
    por_severidade = resumo.get("por_severidade") or {}
    drift = alertas.get("drift") or {}
    return {
        "execucoes": len(execucoes),
        "taxa_sucesso": taxa_sucesso(execucoes),
        "sucessos": por_status.get(STATUS_SUCESSO, 0),
        "rejeicoes": por_status.get(STATUS_REJEICAO, 0),
        "falhas": por_status.get(STATUS_FALHA, 0),
        "incompletas": len(painel["resumo_execucoes"]["incompletas"]),
        "alertas_total": resumo.get("total"),
        "alertas_warning": por_severidade.get(SEVERIDADE_WARNING),
        "alertas_critical": por_severidade.get(SEVERIDADE_CRITICAL),
        "dataset_drift": drift.get("dataset_drift"),
        "variaveis_com_drift": drift.get("variaveis_com_drift"),
    }


def medidas_de_saude(saude: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Reune as medidas de saude por execucao e os agregados, com origem e escopo."""
    linhas: list[dict[str, Any]] = []
    for execucao in (saude or {}).get("execucoes", []) or []:
        if not isinstance(execucao, dict):
            continue
        for medida in execucao.get("metricas", []) or []:
            if isinstance(medida, dict):
                linhas.append(
                    {
                        "origem": execucao.get("run_id"),
                        "escopo": CATEGORIA_EXECUCAO,
                        "id": medida.get("id"),
                        "categoria": medida.get("categoria"),
                        "valor": medida.get("valor"),
                        "unidade": medida.get("unidade"),
                        "situacao": medida.get("situacao"),
                    }
                )
    for medida in (saude or {}).get("agregados", []) or []:
        if isinstance(medida, dict):
            linhas.append(
                {
                    "origem": "agregado",
                    "escopo": CATEGORIA_AGREGADO,
                    "id": medida.get("id"),
                    "categoria": medida.get("categoria"),
                    "valor": medida.get("valor"),
                    "unidade": medida.get("unidade"),
                    "situacao": medida.get("situacao"),
                }
            )
    return linhas
