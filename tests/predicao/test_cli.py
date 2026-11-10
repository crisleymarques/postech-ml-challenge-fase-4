"""CLI da predicao executada em processo separado."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .apoio import NUMERICAS, gerar_lote, montar_config, preparar_manifestos

RAIZ = Path(__file__).resolve().parents[2]


def executar_cli(*argumentos: str) -> subprocess.CompletedProcess:
    ambiente = os.environ.copy()
    ambiente["PYTHONPATH"] = str(RAIZ / "src")
    return subprocess.run(
        [sys.executable, "-m", "monitoramento.predicao.cli", *argumentos],
        cwd=RAIZ, env=ambiente, capture_output=True, text=True, check=False,
    )


def test_cli_expoe_argumentos_existentes() -> None:
    resultado = executar_cli("--help")

    assert resultado.returncode == 0
    for argumento in ("--config", "--input-manifest", "--run-id", "--output-dir", "--analysis-dir"):
        assert argumento in resultado.stdout


def test_cli_carrega_modelo_e_publica_predicoes(tmp_path: Path) -> None:
    manifestos = preparar_manifestos(tmp_path)
    treino = gerar_lote(com_alvo=True)
    pipeline = LogisticRegression(max_iter=200).fit(treino[NUMERICAS], treino["loan_status"])
    joblib.dump(pipeline, tmp_path / "pipeline.pkl")
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["features"] = NUMERICAS
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    for item in manifestos[0]["outputs"]:
        item["sha256"] = sha256_arquivo(Path(item["path"]))
    (tmp_path / "man-treino.json").write_text(json.dumps(manifestos[0]))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(montar_config()))
    output = tmp_path / "run"

    resultado = executar_cli(
        "--config", str(config), "--input-manifest", str(tmp_path / "man-treino.json"),
        "--input-manifest", str(tmp_path / "man-simulacao.json"), "--run-id", "cli-predicao",
        "--output-dir", str(output), "--analysis-dir", str(tmp_path / "analysis"),
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["status"] == "success"
    assert manifesto["outputs"][0]["row_count"] == len(gerar_lote())
