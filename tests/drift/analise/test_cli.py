"""CLI da analise estatistica executada em processo separado."""

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
        [sys.executable, "-m", "monitoramento.drift.analise.cli", *argumentos],
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


def test_cli_publica_analise_estatistica(tmp_path: Path) -> None:
    preparar_manifestos(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(montar_config()), encoding="utf-8")
    output = tmp_path / "run"

    resultado = executar_cli(
        "--config",
        str(config),
        "--input-manifest",
        str(tmp_path / "man-treino.json"),
        "--input-manifest",
        str(tmp_path / "man-simulacao.json"),
        "--run-id",
        "cli-analise",
        "--output-dir",
        str(output),
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["status"] == "success"
    assert manifesto["stage"] == "analise"
    documento = json.loads((output / "analise.json").read_text())
    assert documento["summary"]["changed_columns"] == 5
    assert set(documento["changed_variables"]["confirmed"]) == {"annual_inc", "dti", "revol_util"}


def test_cli_manifesto_ausente_registra_falha(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(montar_config()))
    output = tmp_path / "run"
    resultado = executar_cli(
        "--config",
        str(config),
        "--input-manifest",
        str(tmp_path / "ausente-a.json"),
        "--input-manifest",
        str(tmp_path / "ausente-b.json"),
        "--output-dir",
        str(output),
    )
    assert resultado.returncode == 1
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["outputs"] == []


def test_cli_config_invalida_nao_inicia_execucao(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{")
    output = tmp_path / "run"
    resultado = executar_cli("--config", str(config), "--output-dir", str(output))
    assert resultado.returncode == 1
    assert "Traceback" not in resultado.stderr
    assert not output.exists()
