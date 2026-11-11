"""Consolidacao das contagens, falhas, alertas e transformacoes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.observabilidade import agora_iso

from .classificacao import (
    CATEGORIAS_CONHECIDAS,
    ORDEM_RESULTADOS,
    RESULTADO_DESCONHECIDO,
    STATUS_INCOMPLETO,
    categoria_da_etapa,
    categorias_da_etapa,
)

CONTRACT_VERSION = "1.0"
EVENTOS_DE_FALHA = frozenset({"rejected", "failed"})


def reunir_falhas(
    execucoes: list[dict[str, Any]], eventos: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Preserva falhas dos manifestos quando falta o evento terminal equivalente."""
    falhas = [
        {
            "run_id": evento.get("run_id"),
            "stage": evento.get("stage"),
            "categoria": categoria_da_etapa(evento.get("stage")),
            "event": evento.get("event"),
            "level": evento.get("level"),
            "error_code": evento.get("error_code"),
            "timestamp": evento.get("timestamp"),
        }
        for evento in eventos
        if evento.get("event") in EVENTOS_DE_FALHA or evento.get("level") == "ERROR"
    ]
    registradas = {(falha["run_id"], falha["stage"], falha["event"]) for falha in falhas}
    for execucao in execucoes:
        status = execucao["status"]
        identidade = (execucao["run_id"], execucao["stage"], status)
        if status not in EVENTOS_DE_FALHA or identidade in registradas:
            continue
        falhas.append(
            {
                "run_id": execucao["run_id"],
                "stage": execucao["stage"],
                "categoria": categoria_da_etapa(execucao["stage"]),
                "event": status,
                "level": "ERROR",
                "error_code": execucao.get("error_code"),
                "timestamp": execucao.get("finished_at"),
            }
        )
        registradas.add(identidade)
    return falhas


def montar_resumo(
    *,
    runs_dir: Path,
    output_dir: Path,
    execucoes: list[dict[str, Any]],
    eventos: list[dict[str, Any]],
) -> dict[str, Any]:
    """Consolida contagens de status, categorias, falhas e alertas."""
    por_status: dict[str, int] = {}
    for execucao in execucoes:
        status = execucao["status"]
        por_status[status] = por_status.get(status, 0) + 1

    por_categoria: dict[str, dict[str, int]] = {
        categoria: {resultado: 0 for resultado in ORDEM_RESULTADOS}
        for categoria in CATEGORIAS_CONHECIDAS
    }
    for evento in eventos:
        resultado = evento.get("resultado", RESULTADO_DESCONHECIDO)
        for categoria in categorias_da_etapa(evento.get("stage")):
            contagem = por_categoria.setdefault(categoria, {valor: 0 for valor in ORDEM_RESULTADOS})
            contagem[resultado] = contagem.get(resultado, 0) + 1

    publicadas = {
        (execucao["run_id"], execucao["stage"])
        for execucao in execucoes
        if execucao["status"] == "success"
    }
    alertas = [
        {
            "run_id": evento.get("run_id"),
            "stage": evento.get("stage"),
            "scenario_id": evento.get("scenario_id"),
            "batch_id": evento.get("batch_id"),
            "timestamp": evento.get("timestamp"),
        }
        for evento in eventos
        if evento.get("event") == "drift_alert"
        and (evento.get("run_id"), evento.get("stage")) in publicadas
    ]
    transformacoes = [
        {
            "run_id": execucao["run_id"],
            "nomes": execucao["transformations"],
            "resultado": execucao["resultado"],
        }
        for execucao in execucoes
        if execucao["transformations"]
    ]

    return {
        "contract_version": CONTRACT_VERSION,
        "generated_at": agora_iso(),
        "runs_dir": runs_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "execucoes": {
            "total": len(execucoes),
            "por_status": por_status,
            "incompletas": [
                execucao["run_id"]
                for execucao in execucoes
                if execucao["status"] == STATUS_INCOMPLETO
            ],
        },
        "eventos": {"total": len(eventos)},
        "por_categoria": por_categoria,
        "transformacoes": transformacoes,
        "falhas": reunir_falhas(execucoes, eventos),
        "alertas": alertas,
    }
