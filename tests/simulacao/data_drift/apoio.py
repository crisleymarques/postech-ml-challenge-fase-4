"""Builders compartilhados dos testes de simulacao."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from monitoramento.infraestrutura import arquivos
from monitoramento.simulacao.data_drift import servico


def montar_config() -> dict:
    """Contrato minimo compativel com as variaveis simuladas."""
    return {
        "contract_version": "1.0",
        "stage": "simulacao",
        "dataset_id": "teste",
        "schema_version": "1.0",
        "scenario_id": "data-drift",
        "base": {"artifact_type": "origem", "path": "origem.csv"},
        "time": {"column": "issue_d", "format": "%b-%Y", "shift_months": 24},
        "target": {
            "column": "loan_status",
            "positive_class": "Charged Off",
            "negative_class": "Fully Paid",
        },
        "drift": {
            "seed": 42,
            "jitter": 0.05,
            "variables": {
                "annual_inc": {"operation": "scale", "factor": 0.8, "clip": [0, 10000000]},
                "dti": {"operation": "shift", "delta": 6.0, "clip": [0, 500]},
                "revol_util": {"operation": "scale", "factor": 1.35, "clip": [0, 1000]},
            },
        },
        "outputs": {"analysis_dir": "data/analysis"},
    }


def gerar_origem(quantidade: int = 40) -> pd.DataFrame:
    """Cria uma base de producao simulada determinista."""
    linhas = []
    for indice in range(quantidade):
        linhas.append(
            {
                "record_id": f"LC{indice + 1:07d}",
                "loan_amnt": "10000.0",
                "term": "36 months",
                "annual_inc": str(100000 + (indice % 5) * 1000),
                "issue_d": "Jul-2017",
                "loan_status": str(indice % 2),
                "dti": str(10 + (indice % 4)),
                "fico_range_low": "700.0",
                "fico_range_high": "704.0",
                "revol_util": "950.0" if indice == 0 else str(50 + (indice % 6) * 100),
                "home_ownership": "RENT",
            }
        )
    return pd.DataFrame(linhas)


def escrever_config(tmp_path: Path, config: dict) -> Path:
    caminho = tmp_path / "simulacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")
    return caminho


def escrever_origem(tmp_path: Path, quantidade: int = 40) -> Path:
    caminho = tmp_path / "origem.csv"
    gerar_origem(quantidade).to_csv(caminho, index=False, encoding="utf-8")
    return caminho


def parametros_geracao(tmp_path: Path) -> dict:
    return {
        "config_path": escrever_config(tmp_path, montar_config()),
        "source_path": escrever_origem(tmp_path),
        "analysis_dir": tmp_path / "analysis",
        "batch_id": "batch-1",
        "run_id": "run-1",
        "output_dir": tmp_path / "runs" / "run-1",
        "seed": 42,
    }


def hashes_diretorio(diretorio: Path) -> dict[str, str]:
    return {
        caminho.relative_to(diretorio).as_posix(): arquivos.sha256_arquivo(caminho)
        for caminho in diretorio.rglob("*")
        if caminho.is_file()
    }


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
    return servico.gerar(**parametros)
