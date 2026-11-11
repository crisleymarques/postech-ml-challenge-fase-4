"""Linha de comando da consolidacao dos logs operacionais."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .servico import gerar

RUNS_DIR_PADRAO = Path("reports/runs")
OUTPUT_DIR_PADRAO = Path("reports/logs")
CODIGO_SUCESSO = 0
CODIGO_ERRO = 1


def imprimir_resumo(resumo: dict[str, Any]) -> None:
    """Imprime um resumo legivel da consolidacao."""
    execucoes = resumo["execucoes"]
    por_status = execucoes["por_status"]
    detalhe = ", ".join(
        f"{status}={quantidade}" for status, quantidade in sorted(por_status.items())
    )
    print(
        f"Log operacional consolidado: {execucoes['total']} execucoes, "
        f"{resumo['eventos']['total']} eventos"
    )
    print(f"  por status: {detalhe or 'nenhuma execucao'}")
    print(f"  falhas: {len(resumo['falhas'])} | alertas: {len(resumo['alertas'])}")
    if execucoes["incompletas"]:
        print(f"  incompletas: {', '.join(execucoes['incompletas'])}")
    print(f"  {resumo['output_dir']}/operacional.jsonl")
    print(f"  {resumo['output_dir']}/resumo.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida os logs operacionais do pipeline.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR_PADRAO,
        help="Diretorio com as execucoes (padrao: reports/runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_PADRAO,
        help="Diretorio de saida do log consolidado (padrao: reports/logs).",
    )
    args = parser.parse_args()

    try:
        resumo = gerar(args.runs_dir, args.output_dir)
    except (FileNotFoundError, ValueError) as erro:
        print(f"Falha ao consolidar os logs: {erro}")
        sys.exit(CODIGO_ERRO)

    imprimir_resumo(resumo)
    sys.exit(CODIGO_SUCESSO)
