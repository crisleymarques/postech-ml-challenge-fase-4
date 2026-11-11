"""Dados e builders dos testes de Concept Drift."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SEMENTE = 42

QUANTIDADE = 100

def montar_config() -> dict:
    """Contrato minimo do cenario de Concept Drift."""
    return {
        "contract_version": "1.0",
        "stage": "simulacao",
        "dataset_id": "teste",
        "schema_version": "1.0",
        "scenario_id": "concept-drift",
        "base": {"artifact_type": "origem", "path": "data/simulation_base.csv"},
        "target": {
            "column": "loan_status",
            "positive_class": "Charged Off",
            "negative_class": "Fully Paid",
            "positive_value": 1,
            "negative_value": 0,
        },
        "concept": {
            "seed": SEMENTE,
            "rationale": "Recessao eleva a inadimplencia do segmento de dti alto.",
            "segment": {"column": "dti", "operator": ">=", "value": 25.0},
            "hazard": {
                "reference_value": 25.0,
                "slope": 0.1,
                "scale": 1.0,
                "max_probability": 0.5,
            },
        },
        "outputs": {"analysis_dir": "data/analysis"},
    }

def gerar_base(quantidade: int = QUANTIDADE) -> pd.DataFrame:
    """Cria uma base sintetica com rotulos alternados e dti crescente."""
    return pd.DataFrame(
        {
            "record_id": [f"LC{indice + 1:07d}" for indice in range(quantidade)],
            "dti": [str(indice % 50) for indice in range(quantidade)],
            "annual_inc": [str(60000 + indice * 100) for indice in range(quantidade)],
            "loan_status": [str(indice % 2) for indice in range(quantidade)],
        }
    )

def escrever_config(tmp_path: Path, config: dict) -> Path:
    caminho = tmp_path / "simulacao_concept.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")
    return caminho
