"""Testes da predicao sobre o lote de producao simulado (issue F2-02)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from monitoramento import predicao as pr

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
        "outputs": {"analysis_dir": "data/analysis"},
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
        "sha256": pr.sha256_arquivo(caminho),
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
    pr.escrever_json(caminho_metadata, montar_metadata())
    caminho_features = tmp_path / "features.csv"
    gerar_lote().to_csv(caminho_features, index=False, encoding="utf-8")

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
        tmp_path / "man-simulacao.json",
        run_id="run-simulacao",
        stage="simulacao",
        dataset_id="teste",
        scenario_id="data-drift",
        batch_id="batch-1",
        schema_version="1.0",
        outputs=[descritor("features", caminho_features, "run-simulacao")],
    )
    return [modelo, lote]


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("batch")
    caminho = tmp_path / "predicao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(pr.ErroConfiguracao, match="Chaves ausentes"):
        pr.carregar_config(caminho)


def test_carregar_config_rejeita_etapas_iguais(tmp_path: Path) -> None:
    config = montar_config()
    config["batch"]["producer_stage"] = "treinamento"
    caminho = tmp_path / "predicao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(pr.ErroConfiguracao, match="devem ser diferentes"):
        pr.carregar_config(caminho)


def test_carregar_manifesto_rejeita_status_nao_sucesso(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.json"
    caminho.write_text(
        json.dumps({"contract_version": "1.0", "status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(pr.ErroPredicao, match="nao concluiu com sucesso"):
        pr.carregar_manifesto(caminho)


def test_separar_manifestos_rejeita_ausente() -> None:
    manifesto = {"stage": "treinamento", "run_id": "r1"}

    with pytest.raises(pr.ErroPredicao, match="Manifestos ausentes"):
        pr.separar_manifestos([manifesto], montar_config())


def test_separar_manifestos_rejeita_duplicado() -> None:
    duplicados = [
        {"stage": "treinamento", "run_id": "r1"},
        {"stage": "treinamento", "run_id": "r2"},
        {"stage": "simulacao", "run_id": "r3"},
    ]

    with pytest.raises(pr.ErroPredicao, match="duplicado"):
        pr.separar_manifestos(duplicados, montar_config())


def test_verificar_artefato_rejeita_hash_divergente(tmp_path: Path) -> None:
    caminho = tmp_path / "artefato.csv"
    caminho.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(pr.ErroPredicao, match="Hash divergente"):
        pr.verificar_artefato({"path": caminho.as_posix(), "sha256": "0" * 64})


def test_prever_rejeita_alvo_no_lote(tmp_path: Path) -> None:
    pipeline = criar_pipeline()
    pipeline.fit(
        gerar_lote(com_alvo=True)[FEATURES],
        pd.to_numeric(gerar_lote(com_alvo=True)["loan_status"]).astype(int),
    )

    with pytest.raises(pr.ErroPredicao, match="nao deve conter o alvo"):
        pr.prever(pipeline, montar_metadata(), gerar_lote(com_alvo=True))


def test_prever_rejeita_features_ausentes() -> None:
    lote = gerar_lote().drop(columns=["dti", "term"])
    pipeline = criar_pipeline()

    with pytest.raises(pr.ErroPredicao, match="Features ausentes"):
        pr.prever(pipeline, montar_metadata(), lote)


def test_prever_rejeita_identificador_repetido(tmp_path: Path) -> None:
    pipeline = criar_pipeline()
    treino = gerar_lote(com_alvo=True)
    pipeline.fit(treino[FEATURES], pd.to_numeric(treino["loan_status"]).astype(int))
    lote = gerar_lote()
    lote.loc[1, "record_id"] = lote.loc[0, "record_id"]

    with pytest.raises(pr.ErroPredicao, match="identificadores repetidos"):
        pr.prever(pipeline, montar_metadata(), lote)


def test_prever_marca_score_indisponivel_sem_probabilidade(tmp_path: Path) -> None:
    lote = gerar_lote()
    pipeline = criar_pipeline(LinearSVC(max_iter=5000))
    treino = gerar_lote(com_alvo=True)
    pipeline.fit(treino[FEATURES], pd.to_numeric(treino["loan_status"]).astype(int))

    predicoes, score_disponivel = pr.prever(pipeline, montar_metadata(), lote)

    assert score_disponivel is False
    assert predicoes["score"].isna().all()
    assert list(predicoes.columns) == list(pr.COLUNAS_SAIDA)


def test_processar_publica_predicoes_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "predicao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-pred"

    codigo = pr.processar(
        caminho_config,
        config,
        manifestos,
        "run-pred",
        output_dir,
        analysis_dir=tmp_path / "analysis",
    )

    assert codigo == pr.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "predicao"
    assert manifesto["status"] == "success"
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["reference_id"] == "ref-teste"
    assert manifesto["scenario_id"] == "data-drift"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["score"]["available"] is True
    assert manifesto["score"]["meaning"] == "probabilidade estimada da classe Charged Off"

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == {"predictions"}
    caminho_predicoes = Path(saidas["predictions"]["path"])
    assert caminho_predicoes.is_file()
    assert pr.sha256_arquivo(caminho_predicoes) == saidas["predictions"]["sha256"]

    predicoes = pd.read_csv(caminho_predicoes)
    assert list(predicoes.columns) == list(pr.COLUNAS_SAIDA)
    assert len(predicoes) == len(gerar_lote())
    assert set(predicoes["prediction"]).issubset({0, 1})
    assert predicoes["record_id"].is_unique

    assert caminho_predicoes.as_posix().endswith("data-drift/batch-1/modelo-teste/predictions.csv")
    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert nomes == {"predicoes_geradas", "taxa_positivos", "score_medio"}

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "completed"]


def test_processar_rejeita_dataset_divergente(tmp_path: Path) -> None:
    config = montar_config()
    config["dataset_id"] = "outro-dataset"
    caminho_config = tmp_path / "predicao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)

    codigo = pr.processar(
        caminho_config,
        config,
        manifestos,
        "run-pred",
        tmp_path / "runs" / "run-pred",
        analysis_dir=tmp_path / "analysis",
    )

    assert codigo == pr.CODIGO_ERRO
    eventos = [
        json.loads(linha)
        for linha in (tmp_path / "runs" / "run-pred" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert eventos[-1]["event"] == "failed"
    assert eventos[-1]["error_code"] == "PREDICAO_INVALIDA"
