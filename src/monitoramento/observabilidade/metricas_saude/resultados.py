"""Resumo e publicacao das metricas de saude."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.arquivos import escrever_json_atomico

from .classificacao import ORDEM_SITUACOES, SITUACAO_ATENCAO, SITUACAO_CRITICO


def referencia(run_id: str | None, medida: dict[str, Any]) -> dict[str, Any]:
    """Resume uma medida fora da faixa esperada para o resumo."""
    return {
        "run_id": run_id,
        "id": medida["id"],
        "valor": medida["valor"],
        "situacao": medida["situacao"],
    }


def montar_resumo(
    execucoes: list[dict[str, Any]],
    agregados: list[dict[str, Any]],
) -> dict[str, Any]:
    """Consolida as contagens de situacao e as medidas fora do esperado."""
    todas: list[tuple[str | None, dict[str, Any]]] = [
        (execucao["run_id"], medida) for execucao in execucoes for medida in execucao["metricas"]
    ]
    todas.extend((None, medida) for medida in agregados)

    por_situacao = {situacao: 0 for situacao in ORDEM_SITUACOES}
    for _, medida in todas:
        situacao = medida["situacao"]
        por_situacao[situacao] = por_situacao.get(situacao, 0) + 1

    return {
        "execucoes": len(execucoes),
        "medidas": len(todas),
        "por_situacao": por_situacao,
        "atencao": [
            referencia(run_id, medida)
            for run_id, medida in todas
            if medida["situacao"] == SITUACAO_ATENCAO
        ],
        "criticos": [
            referencia(run_id, medida)
            for run_id, medida in todas
            if medida["situacao"] == SITUACAO_CRITICO
        ],
    }


def gravar_saude(caminho: Path, saude: dict[str, Any]) -> None:
    """Publica o JSON completo, preservando o relatorio anterior se houver falha."""
    escrever_json_atomico(caminho, saude)
