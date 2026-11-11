"""Interface de linha de comando da predicao."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import CODIGO_ERRO, ErroConfiguracao, carregar_config
from .servico import processar


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera predicoes sobre o lote de producao.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/predicao.json"), help="JSON da etapa."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        default=None,
        help="Manifesto de entrada (treinamento, simulacao e validacao). Repetivel.",
    )
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    parser.add_argument(
        "--analysis-dir", type=Path, default=None, help="Diretorio raiz do conjunto simulado."
    )
    args = parser.parse_args()

    try:
        config = carregar_config(args.config)
    except ErroConfiguracao as erro:
        print(f"Falha na predicao: {erro}")
        sys.exit(CODIGO_ERRO)

    entradas = args.input_manifest or []
    if len(entradas) != 3:
        print("Informe tres manifestos: treinamento, simulacao e validacao de producao.")
        sys.exit(CODIGO_ERRO)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id
    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(
        processar(
            args.config,
            config,
            entradas,
            run_id,
            output_dir,
            analysis_dir=args.analysis_dir,
        )
    )


if __name__ == "__main__":
    main()
