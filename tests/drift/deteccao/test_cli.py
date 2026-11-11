"""CLI de deteccao em processo separado."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .apoio import montar_config, preparar_manifestos

RAIZ = Path(__file__).resolve().parents[3]


def executar_cli(*argumentos: str) -> subprocess.CompletedProcess:
    ambiente = os.environ.copy()
    ambiente["PYTHONPATH"] = str(RAIZ / "src")
    return subprocess.run(
        [sys.executable, "-m", "monitoramento.drift.deteccao.cli", *argumentos],
        cwd=RAIZ,
        env=ambiente,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_expoe_argumentos_existentes() -> None:
    resultado = executar_cli("--help")

    assert resultado.returncode == 0, resultado.stderr
    for argumento in ("--config", "--input-manifest", "--run-id", "--output-dir"):
        assert argumento in resultado.stdout


def test_cli_publica_resultados_e_html(tmp_path: Path) -> None:
    preparar_manifestos(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(montar_config()))
    output = tmp_path / "run"

    resultado = executar_cli(
        "--config",
        str(config),
        "--input-manifest",
        str(tmp_path / "man-treino.json"),
        "--input-manifest",
        str(tmp_path / "man-simulacao.json"),
        "--run-id",
        "cli-deteccao",
        "--output-dir",
        str(output),
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["status"] == "success"
    assert (output / "drift.html").stat().st_size > 0
    documento = json.loads((output / "drift.json").read_text())
    assert documento["summary"]["dataset_drift"] is True
