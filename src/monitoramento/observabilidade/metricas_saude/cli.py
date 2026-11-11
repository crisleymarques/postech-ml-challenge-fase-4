"""Linha de comando da coleta de metricas de saude."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .servico import gerar

RUNS_DIR_PADRAO = Path("reports/runs")

OUTPUT_DIR_PADRAO = Path("reports/saude")

CATALOGO_PADRAO = Path("configs/metricas_saude.json")

CODIGO_SUCESSO = 0

CODIGO_ERRO = 1


def imprimir_saude(saude: dict[str, Any]) -> None:
    """Imprime um resumo legivel da saude operacional."""
    resumo = saude["resumo"]
    situacoes = ", ".join(
        f"{situacao}={quantidade}"
        for situacao, quantidade in resumo["por_situacao"].items()
        if quantidade
    )
    print(f"Saude operacional: {resumo['execucoes']} execucoes, {resumo['medidas']} medidas")
    print(f"  situacoes: {situacoes or 'nenhuma'}")
    print(f"  atencao: {len(resumo['atencao'])} | criticos: {len(resumo['criticos'])}")
    for item in resumo["criticos"]:
        origem = item["run_id"] or "agregado"
        print(f"    critico: {item['id']} = {item['valor']} ({origem})")
    print(f"  {saude['output_dir']}/saude.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coleta as metricas de saude operacional do pipeline."
    )
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
        help="Diretorio de saida do relatorio (padrao: reports/saude).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=CATALOGO_PADRAO,
        help="Catalogo de metricas (padrao: configs/metricas_saude.json).",
    )
    args = parser.parse_args()

    try:
        saude = gerar(args.runs_dir, args.output_dir, args.catalog)
    except (FileNotFoundError, ValueError) as erro:
        print(f"Falha ao coletar as metricas de saude: {erro}")
        sys.exit(CODIGO_ERRO)

    imprimir_saude(saude)
    sys.exit(CODIGO_SUCESSO)
