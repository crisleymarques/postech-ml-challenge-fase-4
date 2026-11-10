"""Linha de comando da deteccao de Data Drift."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import CODIGO_ERRO, ErroConfiguracao, ErroDeteccao, carregar_config
from .entradas import carregar_manifesto
from .servico import processar


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta Data Drift entre referencia e producao.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/deteccao.json"), help="JSON da etapa."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        default=None,
        help="Manifesto de entrada (treinamento e simulacao). Repetivel.",
    )
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    args = parser.parse_args()

    try:
        config = carregar_config(args.config)
    except ErroConfiguracao as erro:
        print(f"Falha na deteccao: {erro}")
        sys.exit(CODIGO_ERRO)

    entradas = args.input_manifest or []
    if len(entradas) < 2:
        print("Informe dois manifestos via --input-manifest (um do treinamento e um da simulacao).")
        sys.exit(CODIGO_ERRO)

    try:
        manifestos = [carregar_manifesto(caminho) for caminho in entradas]
    except ErroDeteccao as erro:
        print(f"Falha na deteccao: {erro}")
        sys.exit(CODIGO_ERRO)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id
    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(processar(args.config, config, manifestos, run_id, output_dir))


if __name__ == "__main__":
    main()
