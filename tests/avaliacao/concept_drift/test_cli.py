"""CLI da avaliacao de Concept Drift em processo separado."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .apoio import NUMERICAS, gerar_lote, montar_config, preparar_manifestos

RAIZ = Path(__file__).resolve().parents[3]


def executar_cli(*argumentos: str) -> subprocess.CompletedProcess:
    ambiente = os.environ.copy()
    ambiente["PYTHONPATH"] = str(RAIZ / "src")
    return subprocess.run(
        [sys.executable, "-m", "monitoramento.avaliacao.concept_drift.cli", *argumentos],
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


@pytest.mark.parametrize("classe_unica", [False, True])
def test_cli_publica_comparacao_dos_conceitos(tmp_path: Path, classe_unica: bool) -> None:
    manifestos = preparar_manifestos(tmp_path, inverter=True)
    if classe_unica:
        for item in manifestos[1]["outputs"]:
            if item["artifact_type"] in ("labels", "labels_reference"):
                base = pd.read_csv(item["path"])
                base["loan_status"] = 0
                base.to_csv(item["path"], index=False)
                item["sha256"] = sha256_arquivo(Path(item["path"]))
        (tmp_path / "man-concept.json").write_text(json.dumps(manifestos[1]))
    treino = gerar_lote()
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
        "--config",
        str(config),
        "--input-manifest",
        str(tmp_path / "man-treino.json"),
        "--input-manifest",
        str(tmp_path / "man-concept.json"),
        "--input-manifest",
        str(tmp_path / "man-validacao.json"),
        "--run-id",
        "cli-avaliacao",
        "--output-dir",
        str(output),
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["status"] == "success"
    assert manifesto["stage"] == "avaliacao"
    documento = json.loads((output / "performance.json").read_text())
    assert documento["comparison"]["degradation_detected"] is (None if classe_unica else True)
