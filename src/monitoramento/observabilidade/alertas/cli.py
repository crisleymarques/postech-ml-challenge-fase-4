"""Cli da integracao de alertas operacionais."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .servico import gerar

SAUDE_PADRAO = Path("reports/saude/saude.json")


LOGS_RESUMO_PADRAO = Path("reports/logs/resumo.json")


CONFIG_PADRAO = Path("configs/alertas.json")


OUTPUT_DIR_PADRAO = Path("reports/alertas")


CODIGO_SUCESSO = 0


CODIGO_ERRO = 1


def imprimir_relatorio(relatorio: dict[str, Any]) -> None:
    """Imprime um resumo legivel dos alertas integrados."""
    resumo = relatorio["resumo"]
    severidades = ", ".join(
        f"{severidade}={quantidade}"
        for severidade, quantidade in resumo["por_severidade"].items()
        if quantidade
    )
    drift = relatorio["drift"]
    print(f"Alertas integrados: {resumo['total']} alertas")
    print(f"  severidades: {severidades or 'nenhuma'}")
    print(f"  por categoria: {resumo['por_categoria'] or 'nenhuma'}")
    print(
        "  drift: "
        f"dataset_drift={drift['dataset_drift']}, "
        f"variaveis_com_drift={drift['variaveis_com_drift']}, "
        f"proporcao_drift={drift['proporcao_drift']}, "
        f"alertas={drift['alertas_total']}"
    )
    for identificador in resumo["criticos"]:
        print(f"    critico: {identificador}")
    print(f"  {relatorio['output_dir']}/alertas.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integra drift, metricas de saude e alertas de observabilidade."
    )
    parser.add_argument(
        "--saude",
        type=Path,
        default=SAUDE_PADRAO,
        help="Relatorio de saude da F3-02 (padrao: reports/saude/saude.json).",
    )
    parser.add_argument(
        "--logs-resumo",
        type=Path,
        default=LOGS_RESUMO_PADRAO,
        help="Resumo dos logs da F3-01 (padrao: reports/logs/resumo.json).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PADRAO,
        help="Configuracao dos criterios de alerta (padrao: configs/alertas.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_PADRAO,
        help="Diretorio de saida do relatorio de alertas (padrao: reports/alertas).",
    )
    args = parser.parse_args()

    try:
        relatorio = gerar(args.saude, args.logs_resumo, args.config, args.output_dir)
    except ValueError as erro:
        print(f"Falha ao integrar os alertas: {erro}")
        sys.exit(CODIGO_ERRO)

    imprimir_relatorio(relatorio)
    sys.exit(CODIGO_SUCESSO)
