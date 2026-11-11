"""Linha de comando da simulacao temporal de Data Drift.

Uso: uv run simular-data-drift --config configs/simulacao.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import CODIGO_ERRO
from .servico import gerar


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula a passagem do tempo com Data Drift.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/simulacao.json"), help="JSON da etapa."
    )
    parser.add_argument(
        "--source", type=Path, default=None, help="Base de origem (sobrepoe a configuracao)."
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=None, help="Diretorio raiz do conjunto simulado."
    )
    parser.add_argument("--batch-id", default=None, help="Identificador do lote simulado.")
    parser.add_argument("--seed", type=int, default=None, help="Semente da simulacao.")
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    batch_id = args.batch_id or f"batch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    output_dir = args.output_dir or Path("reports/runs") / run_id

    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(
        gerar(
            config_path=args.config,
            source_path=args.source,
            analysis_dir=args.analysis_dir,
            batch_id=batch_id,
            run_id=run_id,
            output_dir=output_dir,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
