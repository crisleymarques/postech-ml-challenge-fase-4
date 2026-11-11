"""Contrato da CLI executada em um processo separado."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .apoio import escrever_config, escrever_origem, montar_config

RAIZ = Path(__file__).resolve().parents[3]


def executar_cli(*argumentos: str) -> subprocess.CompletedProcess:
    ambiente = os.environ.copy()
    ambiente["PYTHONPATH"] = str(RAIZ / "src")
    return subprocess.run(
        [sys.executable, "-m", "monitoramento.simulacao.data_drift.cli", *argumentos],
        cwd=RAIZ,
        env=ambiente,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_expoe_argumentos_existentes() -> None:
    resultado = executar_cli("--help")

    assert resultado.returncode == 0
    for argumento in (
        "--config",
        "--source",
        "--analysis-dir",
        "--batch-id",
        "--seed",
        "--run-id",
        "--output-dir",
    ):
        assert argumento in resultado.stdout


def test_cli_publica_lote_e_preserva_codigo_de_colisao(tmp_path: Path) -> None:
    config = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)
    run = tmp_path / "run"
    argumentos = (
        "--config",
        str(config),
        "--source",
        str(origem),
        "--analysis-dir",
        str(tmp_path / "analysis"),
        "--batch-id",
        "cli-batch",
        "--run-id",
        "cli-run",
        "--output-dir",
        str(run),
    )

    sucesso = executar_cli(*argumentos)

    assert sucesso.returncode == 0, sucesso.stderr
    manifesto = json.loads((run / "manifest.json").read_text())
    assert manifesto["status"] == "success"
    assert manifesto["batch_id"] == "cli-batch"
    assert manifesto["variaveis_alteradas"] == ["annual_inc", "dti", "revol_util"]
    antes = (run / "manifest.json").read_bytes()

    colisao = executar_cli(*argumentos)

    assert colisao.returncode == 1
    assert (run / "manifest.json").read_bytes() == antes
