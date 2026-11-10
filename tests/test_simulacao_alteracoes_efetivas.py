"""Regressoes para cenarios temporais e distribuicoes realmente alteradas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento import simulacao_drift as sd


def configurar(tmp_path: Path) -> tuple[dict, dict]:
    """Prepara duas features alteradas e uma terceira sem efeito."""
    config = {
        "contract_version": "1.0",
        "stage": "simulacao",
        "dataset_id": "teste",
        "schema_version": "1.0",
        "scenario_id": "data-drift",
        "base": {"path": str(tmp_path / "origem.csv")},
        "time": {"column": "issue_d", "format": "%b-%Y", "shift_months": 24},
        "target": {"column": "loan_status"},
        "drift": {
            "seed": 42,
            "jitter": 0,
            "variables": {
                "annual_inc": {"operation": "scale", "factor": 0.8, "clip": [0, 10000000]},
                "dti": {"operation": "shift", "delta": 6, "clip": [0, 500]},
                "revol_util": {"operation": "scale", "factor": 1, "clip": [0, 1000]},
            },
        },
        "outputs": {"analysis_dir": str(tmp_path / "analysis")},
    }
    pd.DataFrame(
        {
            "record_id": ["LC0000001", "LC0000002"],
            "issue_d": ["Jul-2017", "Dec-2017"],
            "loan_status": [0, 1],
            "annual_inc": [100000, 120000],
            "dti": [10, 20],
            "revol_util": [30, 40],
        }
    ).to_csv(config["base"]["path"], index=False)
    parametros = {
        "config_path": tmp_path / "config.json",
        "source_path": None,
        "analysis_dir": None,
        "batch_id": "batch-1",
        "run_id": "run-1",
        "output_dir": tmp_path / "runs" / "run-1",
        "seed": None,
    }
    return config, parametros


def executar(config: dict, parametros: dict) -> int:
    parametros["config_path"].write_text(json.dumps(config), encoding="utf-8")
    return sd.gerar(**parametros)


@pytest.mark.parametrize("meses", [0, -1, 1.5, 24.0, "24", True])
def test_rejeita_deslocamento_sem_avanco_inteiro(tmp_path: Path, meses: object) -> None:
    config, parametros = configurar(tmp_path)
    config["time"]["shift_months"] = meses

    assert executar(config, parametros) == sd.CODIGO_ERRO
    assert not parametros["output_dir"].exists()
    assert not Path(config["outputs"]["analysis_dir"]).exists()


@pytest.mark.parametrize("cenario", ["neutro", "uma_feature", "zeros", "limite", "ausentes"])
def test_rejeita_menos_de_duas_distribuicoes_alteradas(tmp_path: Path, cenario: str) -> None:
    config, parametros = configurar(tmp_path)
    variaveis = config["drift"]["variables"]
    if cenario == "neutro":
        variaveis["annual_inc"]["factor"] = 1
        variaveis["dti"]["delta"] = 0
    elif cenario == "uma_feature":
        variaveis["dti"]["delta"] = 0
    else:
        origem = Path(config["base"]["path"])
        base = pd.read_csv(origem)
        if cenario == "zeros":
            base["annual_inc"] = 0
        elif cenario == "limite":
            base["dti"] = 500
        else:
            base["annual_inc"] = ""
        base.to_csv(origem, index=False)
    antes = sd.sha256_arquivo(Path(config["base"]["path"]))

    assert executar(config, parametros) == sd.CODIGO_ERRO

    run_dir = parametros["output_dir"]
    manifesto = json.loads((run_dir / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["outputs"] == []
    assert manifesto["variaveis_alteradas"] == []
    assert manifesto["error_code"] == "ERRO_ENTRADA"
    assert json.loads((run_dir / "metrics.json").read_text())["measurements"] == []
    eventos = [json.loads(linha) for linha in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [evento["event"] for evento in eventos] == ["started", "failed"]
    assert not list(Path(config["outputs"]["analysis_dir"]).rglob("*.csv"))
    assert sd.sha256_arquivo(Path(config["base"]["path"])) == antes


def test_reordenar_valores_nao_altera_distribuicao(tmp_path: Path) -> None:
    config, parametros = configurar(tmp_path)
    origem = Path(config["base"]["path"])
    base = pd.read_csv(origem)
    base["dti"] = [-1, 1]
    base.to_csv(origem, index=False)
    config["drift"]["variables"]["dti"] = {
        "operation": "scale", "factor": -1, "clip": [-100, 100]
    }

    assert executar(config, parametros) == sd.CODIGO_ERRO
    manifesto = json.loads((parametros["output_dir"] / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["outputs"] == []


@pytest.mark.parametrize("terceira_ausente", [False, True])
def test_sucesso_informa_apenas_distribuicoes_alteradas(
    tmp_path: Path, terceira_ausente: bool
) -> None:
    config, parametros = configurar(tmp_path)
    if terceira_ausente:
        origem = Path(config["base"]["path"])
        base = pd.read_csv(origem)
        base["revol_util"] = ""
        base.to_csv(origem, index=False)

    assert executar(config, parametros) == sd.CODIGO_SUCESSO

    run_dir = parametros["output_dir"]
    manifesto = json.loads((run_dir / "manifest.json").read_text())
    assert manifesto["status"] == "success"
    assert manifesto["variaveis_alteradas"] == ["annual_inc", "dti"]
    assert set(manifesto["parametros"]["variables"]) == {"annual_inc", "dti"}
    assert len(manifesto["transformacoes"]) == 3
    metricas = json.loads((run_dir / "metrics.json").read_text())["measurements"]
    assert next(m["value"] for m in metricas if m["name"] == "variaveis_alteradas") == 2
    features = next(item for item in manifesto["outputs"] if item["artifact_type"] == "features")
    base_simulada = pd.read_csv(features["path"], dtype=str, na_filter=False)
    assert list(base_simulada["issue_d"]) == ["Jul-2019", "Dec-2019"]
