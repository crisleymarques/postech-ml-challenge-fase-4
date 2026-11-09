"""Testes da avaliacao de desempenho sob Concept Drift (issue F2-05)."""

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

from monitoramento import avaliacao_concept_drift as ac

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
        "sha256": ac.sha256_arquivo(caminho),
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
    ac.escrever_json(caminho_metadata, montar_metadata())

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


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("metrics")
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ac.ErroConfiguracao, match="Chaves ausentes"):
        ac.carregar_config(caminho)


def test_carregar_config_rejeita_etapas_iguais(tmp_path: Path) -> None:
    config = montar_config()
    config["batch"]["producer_stage"] = "treinamento"
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ac.ErroConfiguracao, match="devem ser diferentes"):
        ac.carregar_config(caminho)


def test_carregar_config_rejeita_metrica_invalida(tmp_path: Path) -> None:
    config = montar_config()
    config["metrics"] = ["accuracy", "xpto"]
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ac.ErroConfiguracao, match="invalidas"):
        ac.carregar_config(caminho)


def test_carregar_config_rejeita_metrica_de_degradacao_fora_da_lista(tmp_path: Path) -> None:
    config = montar_config()
    config["metrics"] = ["accuracy"]
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ac.ErroConfiguracao, match="degradacao"):
        ac.carregar_config(caminho)


def test_carregar_config_rejeita_min_delta_nao_positivo(tmp_path: Path) -> None:
    config = montar_config()
    config["degradation"]["min_delta"] = 0
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ac.ErroConfiguracao, match="min_delta"):
        ac.carregar_config(caminho)


def test_carregar_config_rejeita_arquivo_ausente(tmp_path: Path) -> None:
    with pytest.raises(ac.ErroConfiguracao, match="nao encontrada"):
        ac.carregar_config(tmp_path / "inexistente.json")


def test_carregar_manifesto_rejeita_status_nao_sucesso(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.json"
    caminho.write_text(
        json.dumps({"contract_version": "1.0", "status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(ac.ErroAvaliacao, match="nao concluiu com sucesso"):
        ac.carregar_manifesto(caminho)


def test_separar_manifestos_rejeita_ausente() -> None:
    manifesto = {"stage": "treinamento", "run_id": "r1"}

    with pytest.raises(ac.ErroAvaliacao, match="Manifestos ausentes"):
        ac.separar_manifestos([manifesto], montar_config())


def test_separar_manifestos_rejeita_duplicado() -> None:
    duplicados = [
        {"stage": "treinamento", "run_id": "r1"},
        {"stage": "treinamento", "run_id": "r2"},
        {"stage": "simulacao", "run_id": "r3"},
    ]

    with pytest.raises(ac.ErroAvaliacao, match="duplicado"):
        ac.separar_manifestos(duplicados, montar_config())


def test_artefato_rejeita_tipo_ausente() -> None:
    with pytest.raises(ac.ErroAvaliacao, match="ausente"):
        ac.artefato({"run_id": "r", "outputs": []}, "model_pipeline")


def test_verificar_artefato_rejeita_hash_divergente(tmp_path: Path) -> None:
    caminho = tmp_path / "artefato.csv"
    caminho.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ac.ErroAvaliacao, match="Hash divergente"):
        ac.verificar_artefato({"path": caminho.as_posix(), "sha256": "0" * 64})


def test_verificar_compatibilidade_rejeita_dataset_divergente() -> None:
    with pytest.raises(ac.ErroAvaliacao, match="dataset_id divergente"):
        ac.verificar_compatibilidade(
            montar_config(),
            {"run_id": "r1", "dataset_id": "outro"},
            {"run_id": "r2", "dataset_id": "teste"},
        )


def test_ler_lote_trata_campo_vazio_como_ausencia(tmp_path: Path) -> None:
    caminho = tmp_path / "features.csv"
    pd.DataFrame({"record_id": ["LC1", "LC2"], "dti": ["10", ""]}).to_csv(
        caminho, index=False, encoding="utf-8"
    )

    lote = ac.ler_lote(caminho)

    assert lote.loc[0, "dti"] == "10"
    assert pd.isna(lote.loc[1, "dti"])


def test_juntar_rotulos_alinhado_pelo_identificador(tmp_path: Path) -> None:
    caminho = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": ["LC2", "LC1"], "loan_status": ["1", "0"]}).to_csv(
        caminho, index=False, encoding="utf-8"
    )

    rotulos = ac.juntar_rotulos(pd.Series(["LC1", "LC2"]), caminho, "loan_status")

    assert rotulos.tolist() == [0, 1]


def test_juntar_rotulos_rejeita_identificadores_repetidos(tmp_path: Path) -> None:
    caminho = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": ["LC1"], "loan_status": ["0"]}).to_csv(
        caminho, index=False, encoding="utf-8"
    )

    with pytest.raises(ac.ErroAvaliacao, match="repetidos"):
        ac.juntar_rotulos(pd.Series(["LC1", "LC1"]), caminho, "loan_status")


def test_juntar_rotulos_rejeita_rotulos_ausentes(tmp_path: Path) -> None:
    caminho = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": ["LC1"], "loan_status": ["0"]}).to_csv(
        caminho, index=False, encoding="utf-8"
    )

    with pytest.raises(ac.ErroAvaliacao, match="Rotulos ausentes"):
        ac.juntar_rotulos(pd.Series(["LC1", "LC2"]), caminho, "loan_status")


def test_calcular_metricas_retorna_valores() -> None:
    alvo = np.array([0, 0, 1, 1])
    predito = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.4, 0.6, 0.9])

    valores = ac.calcular_metricas(alvo, predito, score, ac.METRICAS)

    assert set(valores) == set(ac.METRICAS)
    assert valores["accuracy"] == 0.5
    assert valores["roc_auc"] == pytest.approx(1.0)


def test_calcular_metricas_sem_classe_positiva_nao_calcula_auc() -> None:
    alvo = np.array([0, 0, 0, 0])
    predito = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.6, 0.9])

    valores = ac.calcular_metricas(alvo, predito, score, ("roc_auc",))

    assert valores["roc_auc"] is None


def test_comparar_detecta_degradacao() -> None:
    resultado = ac.comparar({"roc_auc": 0.70}, {"roc_auc": 0.65}, "roc_auc", 0.02)

    assert resultado["degradation_detected"] is True
    assert resultado["metric_delta"] == pytest.approx(-0.05)


def test_comparar_sem_degradacao_relevante() -> None:
    resultado = ac.comparar({"roc_auc": 0.70}, {"roc_auc": 0.69}, "roc_auc", 0.02)

    assert resultado["degradation_detected"] is False


def test_montar_conclusoes_contrasta_data_e_concept_drift() -> None:
    referencia = {"roc_auc": 0.70, "accuracy": 0.70}
    alterado = {"roc_auc": 0.60, "accuracy": 0.60}
    comparacao = ac.comparar(referencia, alterado, "roc_auc", 0.02)

    conclusoes = ac.montar_conclusoes(
        referencia, alterado, comparacao, {"description": "recessao no segmento"}
    )
    texto = " ".join(conclusoes)

    assert "P(y|X)" in texto
    assert "P(X)" in texto
    assert "recessao no segmento" in texto


def test_montar_medidas_registra_variacao() -> None:
    referencia = {"roc_auc": 0.70}
    alterado = {"roc_auc": 0.60}
    comparacao = ac.comparar(referencia, alterado, "roc_auc", 0.02)

    medidas = ac.montar_medidas(comparacao, referencia, alterado)
    nomes = {item["name"] for item in medidas}

    assert nomes == {
        "roc_auc_referencia",
        "roc_auc_concept_drift",
        "variacao_roc_auc",
        "degradacao_detectada",
    }


def test_processar_publica_performance_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "avaliacao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path, inverter=True)
    output_dir = tmp_path / "runs" / "run-aval"

    codigo = ac.processar(caminho_config, config, manifestos, "run-aval", output_dir)

    assert codigo == ac.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "avaliacao"
    assert manifesto["status"] == "success"
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["reference_id"] == "ref-teste"
    assert manifesto["scenario_id"] == "concept-drift"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["comparison"]["degradation_detected"] is True

    documento = json.loads((output_dir / "performance.json").read_text(encoding="utf-8"))
    assert documento["comparison"]["metric"] == "roc_auc"
    assert (
        documento["contrast"]["reference"]["metrics"]["roc_auc"]
        > documento["contrast"]["concept_drift"]["metrics"]["roc_auc"]
    )
    texto = " ".join(documento["interpretation"]["conclusions"])
    assert "Data Drift" in texto
    assert "Concept Drift" in texto

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert "roc_auc_referencia" in nomes
    assert "degradacao_detectada" in nomes

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "drift_alert", "completed"]
    alerta = next(evento for evento in eventos if evento["event"] == "drift_alert")
    assert alerta["level"] == "WARNING"


def test_processar_sem_degradacao_nao_emite_alerta(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "avaliacao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path, inverter=False)
    output_dir = tmp_path / "runs" / "run-sem-alerta"

    codigo = ac.processar(caminho_config, config, manifestos, "run-sem-alerta", output_dir)

    assert codigo == ac.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["comparison"]["degradation_detected"] is False

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "completed"]


def test_processar_rejeita_dataset_divergente(tmp_path: Path) -> None:
    config = montar_config()
    config["dataset_id"] = "outro-dataset"
    caminho_config = tmp_path / "avaliacao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-aval"

    codigo = ac.processar(caminho_config, config, manifestos, "run-aval", output_dir)

    assert codigo == ac.CODIGO_ERRO
    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert eventos[-1]["event"] == "failed"
    assert eventos[-1]["error_code"] == "AVALIACAO_INVALIDA"
