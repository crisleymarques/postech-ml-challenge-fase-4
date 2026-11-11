"""Testes da predicao sobre o lote de producao simulado (issue F2-02)."""

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


def montar_config() -> dict:
    """Contrato minimo da etapa de predicao."""
    return {
        "contract_version": "1.0",
        "stage": "predicao",
        "dataset_id": "teste",
        "identifier": "record_id",
        "model": {
            "producer_stage": "treinamento",
            "pipeline_artifact": "model_pipeline",
            "metadata_artifact": "model_metadata",
        },
        "batch": {"producer_stage": "simulacao", "features_artifact": "features"},
        "validation": {"producer_stage": "validacao"},
        "outputs": {"analysis_dir": "data/analysis"},
    }


def montar_metadata() -> dict:
    """Metadados compativeis com as features sinteticas."""
    return {
        "contract_version": "1.0",
        "schema_version": "1.0",
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


def criar_pipeline(estimador: object | None = None) -> Pipeline:
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
            ("clf", estimador or LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )


def gerar_lote(quantidade: int = 12, com_alvo: bool = False) -> pd.DataFrame:
    """Cria um lote sintetico de producao."""
    linhas = []
    for indice in range(quantidade):
        linha = {
            "record_id": f"LC{indice + 1:07d}",
            "annual_inc": str(60000 + indice * 1000),
            "dti": str(10 + indice % 5),
            "revol_util": str(30 + indice),
            "term": "36 months" if indice % 2 else "60 months",
            "home_ownership": "RENT" if indice % 3 else "OWN",
        }
        if com_alvo:
            linha["loan_status"] = str(indice % 2)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def treinar_pipeline(tmp_path: Path, estimador: object | None = None) -> Path:
    """Persiste um pipeline ajustado em disco."""
    treino = gerar_lote(com_alvo=True)
    rotulos = pd.to_numeric(treino["loan_status"], errors="raise").astype(int)
    pipeline = criar_pipeline(estimador)
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


def preparar_manifestos(tmp_path: Path, estimador: object | None = None) -> list[dict]:
    """Monta os manifestos de treinamento e simulacao com hashes reais."""
    caminho_pipeline = treinar_pipeline(tmp_path, estimador)
    caminho_metadata = tmp_path / "metadata.json"
    arquivos.escrever_json(caminho_metadata, montar_metadata())
    caminho_features = tmp_path / "features.csv"
    gerar_lote().to_csv(caminho_features, index=False, encoding="utf-8")

    modelo = escrever_manifesto(
        tmp_path / "man-treino.json",
        run_id="run-treino",
        stage="treinamento",
        schema_version="1.0",
        dataset_id="teste",
        model_id="modelo-teste",
        reference_id="ref-teste",
        outputs=[
            descritor("model_pipeline", caminho_pipeline, "run-treino"),
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
        outputs=[descritor("features", caminho_features, "run-simulacao")],
    )
    features = lote["outputs"][0]
    validacao = escrever_manifesto(
        tmp_path / "man-validacao.json",
        stage="validacao",
        run_id="run-validacao",
        dataset_id="teste",
        schema_version="1.0",
        scenario_id="data-drift",
        batch_id="batch-1",
        inputs=[dict(features)],
        outputs=[],
        quality={
            "approved": True,
            "dataset_id": "teste",
            "schema_version": "1.0",
            "artifacts": [
                {"artifact_type": "features", "sha256": features["sha256"], "approved": True}
            ],
        },
    )
    return [modelo, lote, validacao]
