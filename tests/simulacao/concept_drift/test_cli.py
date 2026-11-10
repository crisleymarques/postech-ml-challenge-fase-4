"""CLI da simulacao de Concept Drift em processo separado."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .apoio import escrever_config, gerar_base, montar_config

RAIZ = Path(__file__).resolve().parents[3]


def executar_cli(*argumentos: str) -> subprocess.CompletedProcess:
    ambiente = os.environ.copy()
    ambiente["PYTHONPATH"] = str(RAIZ / "src")
    return subprocess.run(
        [sys.executable, "-m", "monitoramento.simulacao.concept_drift.cli", *argumentos],
        cwd=RAIZ, env=ambiente, capture_output=True, text=True, check=False,
    )


def test_cli_expoe_argumentos_existentes() -> None:
    resultado = executar_cli("--help")

    assert resultado.returncode == 0, resultado.stderr
    for argumento in (
        "--config", "--source", "--analysis-dir", "--batch-id", "--seed", "--run-id", "--output-dir"
    ):
        assert argumento in resultado.stdout
    assert "--input-manifest" not in resultado.stdout


def test_cli_publica_features_e_dois_conjuntos_de_rotulos(tmp_path: Path) -> None:
    origem = tmp_path / "origem.csv"
    base = gerar_base()
    base.to_csv(origem, index=False)
    config = escrever_config(tmp_path, montar_config())
    output = tmp_path / "run"

    resultado = executar_cli(
        "--config", str(config), "--source", str(origem),
        "--analysis-dir", str(tmp_path / "analysis"), "--batch-id", "cli-batch",
        "--run-id", "cli-concept", "--output-dir", str(output),
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["features_unchanged"] is True
    assert {item["artifact_type"] for item in manifesto["outputs"]} == {
        "features", "labels", "labels_reference"
    }
    assert manifesto["concept_change"]["flipped_rows"] > 0
