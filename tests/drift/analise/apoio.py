"""Dados e manifestos sinteticos compartilhados pelos testes de analise."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from monitoramento.infraestrutura.arquivos import escrever_json, sha256_arquivo

NUMERICAS = ["annual_inc", "dti", "revol_util"]
CATEGORICAS = ["term", "home_ownership"]
FEATURES = NUMERICAS + CATEGORICAS
SEMENTE = 42
QUANTIDADE = 400


def montar_config() -> dict:
    """Contrato minimo da etapa de analise."""
    return {
        "contract_version": "1.0",
        "stage": "analise",
        "dataset_id": "teste",
        "identifier": "record_id",
        "reference": {
            "producer_stage": "treinamento",
            "features_artifact": "reference_features",
            "metadata_artifact": "model_metadata",
        },
        "batch": {"producer_stage": "simulacao", "features_artifact": "features"},
        "method": {
            "name": "psi_ks",
            "bins": 10,
            "psi_moderate": 0.1,
            "psi_threshold": 0.25,
            "ks_p_value_threshold": 0.05,
            "ks_statistic_threshold": 0.1,
        },
    }


def montar_metadata() -> dict:
    """Metadados compativeis com as features sinteticas."""
    return {
        "contract_version": "1.0",
        "model_id": "modelo-teste",
        "reference_id": "ref-teste",
        "dataset_id": "teste",
        "features": FEATURES,
        "numeric_features": NUMERICAS,
        "categorical_features": CATEGORICAS,
    }


def gerar_features(deslocado: bool, quantidade: int = QUANTIDADE) -> pd.DataFrame:
    """Cria features sinteticas de referencia ou de producao com mudanca clara."""
    rng = np.random.default_rng(SEMENTE + (1 if deslocado else 0))
    if deslocado:
        annual_inc = rng.normal(120000, 5000, quantidade)
        dti = rng.normal(40, 2, quantidade)
        revol_util = rng.normal(80, 5, quantidade)
        term = rng.choice(["36 months", "60 months"], quantidade, p=[0.2, 0.8])
        home = rng.choice(["RENT", "OWN", "MORTGAGE"], quantidade, p=[0.7, 0.2, 0.1])
    else:
        annual_inc = rng.normal(60000, 5000, quantidade)
        dti = rng.normal(10, 2, quantidade)
        revol_util = rng.normal(30, 5, quantidade)
        term = rng.choice(["36 months", "60 months"], quantidade, p=[0.8, 0.2])
        home = rng.choice(["RENT", "OWN", "MORTGAGE"], quantidade, p=[0.2, 0.3, 0.5])

    return pd.DataFrame(
        {
            "record_id": [f"LC{indice:07d}" for indice in range(quantidade)],
            "annual_inc": annual_inc,
            "dti": dti,
            "revol_util": revol_util,
            "term": term,
            "home_ownership": home,
        }
    )


def descritor(tipo: str, caminho: Path, run_id: str) -> dict:
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }


def escrever_manifesto(caminho: Path, **campos: object) -> dict:
    conteudo = {"contract_version": "1.0", "status": "success"}
    conteudo.update(campos)
    caminho.write_text(json.dumps(conteudo), encoding="utf-8")
    return conteudo


def preparar_manifestos(tmp_path: Path) -> list[dict]:
    """Monta os manifestos de treinamento e simulacao com hashes reais."""
    caminho_ref = tmp_path / "reference_features.csv"
    gerar_features(deslocado=False).to_csv(caminho_ref, index=False, encoding="utf-8")
    caminho_lote = tmp_path / "features.csv"
    gerar_features(deslocado=True).to_csv(caminho_lote, index=False, encoding="utf-8")
    caminho_metadata = tmp_path / "metadata.json"
    escrever_json(caminho_metadata, montar_metadata())

    treino = escrever_manifesto(
        tmp_path / "man-treino.json",
        run_id="run-treino",
        stage="treinamento",
        dataset_id="teste",
        model_id="modelo-teste",
        reference_id="ref-teste",
        outputs=[
            descritor("reference_features", caminho_ref, "run-treino"),
            descritor("model_metadata", caminho_metadata, "run-treino"),
        ],
    )
    lote = escrever_manifesto(
        tmp_path / "man-simulacao.json",
        run_id="run-simulacao",
        stage="simulacao",
        dataset_id="teste",
        scenario_id="data-drift",
        batch_id="batch-1",
        schema_version="1.0",
        variaveis_alteradas=["annual_inc", "dti", "revol_util"],
        outputs=[descritor("features", caminho_lote, "run-simulacao")],
    )
    return [treino, lote]
