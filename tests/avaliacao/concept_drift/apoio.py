"""Dados e builders dos testes de Concept Drift."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from monitoramento.infraestrutura import arquivos

NUMERICAS = ["annual_inc", "dti", "revol_util"]

CATEGORICAS = ["term", "home_ownership"]

FEATURES = NUMERICAS + CATEGORICAS

SEMENTE = 42

QUANTIDADE = 24

def montar_config() -> dict:
    """Contrato minimo da etapa de avaliacao."""
    return {
        "contract_version": "1.0",
        "stage": "avaliacao",
        "dataset_id": "teste",
        "identifier": "record_id",
        "model": {
            "producer_stage": "treinamento",
            "pipeline_artifact": "model_pipeline",
            "metadata_artifact": "model_metadata",
        },
        "batch": {
            "producer_stage": "simulacao",
            "features_artifact": "features",
            "labels_artifact": "labels",
            "reference_labels_artifact": "labels_reference",
        },
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "degradation": {"metric": "roc_auc", "min_delta": 0.02},
    }

def montar_metadata() -> dict:
    """Metadados compativeis com as features sinteticas."""
    return {
        "contract_version": "1.0",
        "model_id": "modelo-teste",
        "reference_id": "ref-teste",
        "dataset_id": "teste",
        "target": {
            "column": "loan_status",
            "positive_class": "Charged Off",
            "negative_class": "Fully Paid",
            "positive_value": 1,
            "negative_value": 0,
        },
        "features": FEATURES,
        "numeric_features": NUMERICAS,
        "categorical_features": CATEGORICAS,
        "prediction_threshold": 0.5,
        "score_meaning": "probabilidade estimada da classe Charged Off",
    }

def _converter_numericas(tabela: object) -> object:
    return pd.DataFrame(tabela).apply(pd.to_numeric, errors="coerce").to_numpy()

def _converter_categoricas(tabela: object) -> object:
    return pd.DataFrame(tabela).replace("", np.nan).to_numpy()

def criar_pipeline() -> Pipeline:
    """Reproduz a estrutura de pre-processamento do treinamento."""
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("converter", FunctionTransformer(_converter_numericas)),
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                NUMERICAS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("converter", FunctionTransformer(_converter_categoricas)),
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAS,
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )

def gerar_lote(quantidade: int = QUANTIDADE, inverter: bool = False) -> pd.DataFrame:
    """Cria um lote sintetico cujo rotulo depende de uma regra aprendivel."""
    linhas = []
    for indice in range(quantidade):
        rotulo = 1 if indice < quantidade // 2 else 0
        if inverter:
            rotulo = 1 - rotulo
        linhas.append(
            {
                "record_id": f"LC{indice + 1:07d}",
                "annual_inc": str(60000 + indice * 1000),
                "dti": str(10 + indice % 5),
                "revol_util": str(30 + indice),
                "term": "36 months" if indice % 2 else "60 months",
                "home_ownership": "RENT" if indice % 3 else "OWN",
                "loan_status": str(rotulo),
            }
        )
    return pd.DataFrame(linhas)

def treinar_pipeline(tmp_path: Path) -> Path:
    """Persiste um pipeline ajustado em disco."""
    treino = gerar_lote()
    rotulos = pd.to_numeric(treino["loan_status"]).astype(int)
    pipeline = criar_pipeline()
    pipeline.fit(treino[FEATURES], rotulos)
    caminho = tmp_path / "pipeline.pkl"
    joblib.dump(pipeline, caminho)
    return caminho

def descritor(tipo: str, caminho: Path, run_id: str) -> dict:
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": arquivos.sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }

def escrever_manifesto(caminho: Path, **campos: object) -> dict:
    conteudo = {"contract_version": "1.0", "status": "success"}
    conteudo.update(campos)
    caminho.write_text(json.dumps(conteudo), encoding="utf-8")
    return conteudo

def preparar_manifestos(tmp_path: Path, inverter: bool = True) -> list[dict]:
    """Monta os manifestos de treinamento e simulacao com hashes reais."""
    caminho_pipeline = treinar_pipeline(tmp_path)
    caminho_metadata = tmp_path / "metadata.json"
    arquivos.escrever_json(caminho_metadata, montar_metadata())

    caminho_features = tmp_path / "features.csv"
    gerar_lote().drop(columns=["loan_status"]).to_csv(
        caminho_features, index=False, encoding="utf-8"
    )
    caminho_labels = tmp_path / "labels.csv"
    gerar_lote(inverter=inverter)[["record_id", "loan_status"]].to_csv(
        caminho_labels, index=False, encoding="utf-8"
    )
    caminho_referencia = tmp_path / "labels_reference.csv"
    gerar_lote()[["record_id", "loan_status"]].to_csv(
        caminho_referencia, index=False, encoding="utf-8"
    )

    modelo = escrever_manifesto(
        tmp_path / "man-treino.json",
        run_id="run-treino",
        stage="treinamento",
        dataset_id="teste",
        model_id="modelo-teste",
        reference_id="ref-teste",
        outputs=[
            descritor("model_pipeline", caminho_pipeline, "run-treino"),
            descritor("model_metadata", caminho_metadata, "run-treino"),
        ],
    )
    lote = escrever_manifesto(
        tmp_path / "man-concept.json",
        run_id="run-concept",
        stage="simulacao",
        dataset_id="teste",
        scenario_id="concept-drift",
        batch_id="batch-1",
        schema_version="1.0",
        concept_change={"description": "Recessao eleva a inadimplencia do segmento."},
        outputs=[
            descritor("features", caminho_features, "run-concept"),
            descritor("labels", caminho_labels, "run-concept"),
            descritor("labels_reference", caminho_referencia, "run-concept"),
        ],
    )
    return [modelo, lote]
