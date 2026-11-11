"""Fluxo real por CLI, com dados sinteticos e modelo produzido pelo treinamento."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from monitoramento.infraestrutura.arquivos import escrever_json, sha256_arquivo

from .apoio import CATEGORICAS, FEATURES, NUMERICAS, gerar_lote, montar_config, montar_metadata

RAIZ = Path(__file__).resolve().parents[2]


def test_fluxo_completo_por_cli(tmp_path: Path) -> None:
    ambiente = {**os.environ, "PYTHONPATH": str(RAIZ / "src")}

    def executar(*argumentos):
        resultado = subprocess.run(
            [sys.executable, *map(str, argumentos)],
            cwd=RAIZ,
            env=ambiente,
            capture_output=True,
            text=True,
            check=False,
        )
        assert resultado.returncode == 0, resultado.stdout + resultado.stderr

    alvo = montar_metadata()["target"]
    entradas = []
    for indice, papel in enumerate(("train", "validation", "test")):
        base = gerar_lote(60, com_alvo=True)
        base["record_id"] = [f"LC{indice * 1000 + i:07d}" for i in range(len(base))]
        caminho = tmp_path / f"{papel}.csv"
        base.to_csv(caminho, index=False)
        entradas.append(
            {
                "artifact_type": papel,
                "path": str(caminho),
                "sha256": sha256_arquivo(caminho),
                "producer_run_id": "prep",
                "row_count": len(base),
                "columns": list(base.columns),
            }
        )
    origem = tmp_path / "preparacao.json"
    escrever_json(
        origem,
        {
            "contract_version": "1.0",
            "stage": "preparacao",
            "status": "success",
            "run_id": "prep",
            "dataset_id": "teste",
            "target": alvo,
            "outputs": entradas,
        },
    )
    contrato = tmp_path / "contrato.json"
    escrever_json(
        contrato,
        {
            "contract_version": "1.0",
            "stage": "validacao",
            "profile": "reference",
            "dataset_id": "teste",
            "schema_version": "1.0",
            "target_contract": alvo,
            "artifacts": ["train", "validation", "test"],
            "rules": {
                "identifier": {"column": "record_id", "pattern": "^LC[0-9]{7}$"},
                "target": {"column": "loan_status", "domain": [0, 1]},
                "not_null": ["record_id", "annual_inc", "term", "loan_status"],
                "unique": ["record_id"],
                "ranges": {"annual_inc": [0, 10000000]},
                "categories": {"term": ["36 months", "60 months"]},
            },
        },
    )
    runs = tmp_path / "runs"
    executar(
        "src/monitoramento/validacao.py",
        "--config",
        contrato,
        "--input-manifest",
        origem,
        "--output-dir",
        runs / "val-ref",
        "--run-id",
        "val-ref",
    )
    treino_config = json.loads((RAIZ / "configs/treinamento.json").read_text())
    treino_config.update(dataset_id="teste", model_id="baseline-integracao", target=alvo)
    treino_config["features"] = {
        "identifier": "record_id",
        "numeric": NUMERICAS,
        "categorical": CATEGORICAS,
    }
    treino_config["outputs"] = {
        "models_dir": str(tmp_path / "models"),
        "reference_dir": str(tmp_path / "reference"),
    }
    treino_path = tmp_path / "treinamento.json"
    escrever_json(treino_path, treino_config)
    executar(
        "src/monitoramento/treinamento.py",
        "--config",
        treino_path,
        "--input-manifest",
        runs / "val-ref/manifest.json",
        "--output-dir",
        runs / "treino",
        "--run-id",
        "treino",
    )
    base_simulada = gerar_lote(50, com_alvo=True)
    base_simulada["record_id"] = [f"LC{3000 + i:07d}" for i in range(len(base_simulada))]
    base_simulada["issue_d"] = "Jul-2017"
    base_path = tmp_path / "simulation_base.csv"
    base_simulada.to_csv(base_path, index=False)
    simulacao_config = json.loads((RAIZ / "configs/simulacao.json").read_text())
    simulacao_config.update(dataset_id="teste", scenario_id="integracao")
    simulacao_config["base"]["path"] = str(base_path)
    simulacao_config["drift"]["variables"] = {
        "annual_inc": {"operation": "scale", "factor": 0.8, "clip": [0, 10000000]},
        "dti": {"operation": "shift", "delta": 6, "clip": [0, 500]},
    }
    simulacao_config["outputs"]["analysis_dir"] = str(tmp_path / "analysis")
    simulacao_path = tmp_path / "simulacao.json"
    escrever_json(simulacao_path, simulacao_config)
    executar(
        "-m",
        "monitoramento.simulacao.data_drift.cli",
        "--config",
        simulacao_path,
        "--batch-id",
        "batch-integracao",
        "--run-id",
        "sim",
        "--output-dir",
        runs / "sim",
    )
    executar(
        "src/monitoramento/validacao.py",
        "--config",
        contrato,
        "--profile",
        "production",
        "--input-manifest",
        runs / "sim/manifest.json",
        "--run-id",
        "val-prod",
        "--output-dir",
        runs / "val-prod",
    )
    predicao_config = tmp_path / "predicao.json"
    config = montar_config()
    config["outputs"]["analysis_dir"] = str(tmp_path / "analysis")
    escrever_json(predicao_config, config)
    executar(
        "-m",
        "monitoramento.predicao.cli",
        "--config",
        predicao_config,
        "--input-manifest",
        runs / "val-prod/manifest.json",
        "--input-manifest",
        runs / "sim/manifest.json",
        "--input-manifest",
        runs / "treino/manifest.json",
        "--run-id",
        "pred",
        "--output-dir",
        runs / "pred",
    )
    treino = json.loads((runs / "treino/manifest.json").read_text())
    manifesto = json.loads((runs / "pred/manifest.json").read_text())
    assert manifesto["status"] == "success"
    assert manifesto["quality_validation_run_id"] == "val-prod"
    for descritor in manifesto["inputs"] + manifesto["outputs"]:
        assert sha256_arquivo(Path(descritor["path"])) == descritor["sha256"]
    predicoes = pd.read_csv(manifesto["outputs"][0]["path"], dtype={"record_id": str})
    sim = json.loads((runs / "sim/manifest.json").read_text())
    features = next(i for i in sim["outputs"] if i["artifact_type"] == "features")
    lote = pd.read_csv(features["path"], dtype=str, na_filter=False)
    pipeline = joblib.load(
        next(i["path"] for i in treino["outputs"] if i["artifact_type"] == "model_pipeline")
    )
    assert predicoes["record_id"].tolist() == lote["record_id"].tolist()
    assert predicoes["prediction"].tolist() == pipeline.predict(lote[FEATURES]).tolist()
    np.testing.assert_allclose(
        predicoes["score"], pipeline.predict_proba(lote[FEATURES])[:, 1], atol=0.000001, rtol=0
    )
    assert set(predicoes["prediction"]).issubset({0, 1})
