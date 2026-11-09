"""Testes da deteccao de Data Drift com Evidently (issue F2-03)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoramento import deteccao as dt

NUMERICAS = ["annual_inc", "dti", "revol_util"]
CATEGORICAS = ["term", "home_ownership"]
FEATURES = NUMERICAS + CATEGORICAS

SEMENTE = 42
QUANTIDADE = 400


def montar_config() -> dict:
    """Contrato minimo da etapa de deteccao."""
    return {
        "contract_version": "1.0",
        "stage": "deteccao",
        "dataset_id": "teste",
        "identifier": "record_id",
        "reference": {
            "producer_stage": "treinamento",
            "features_artifact": "reference_features",
            "metadata_artifact": "model_metadata",
        },
        "batch": {"producer_stage": "simulacao", "features_artifact": "features"},
        "method": {
            "name": "evidently",
            "preset": "DataDriftPreset",
            "p_value_threshold": 0.05,
            "distance_threshold": 0.1,
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
    """Cria features sinteticas de referencia ou de producao com drift claro."""
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
        "sha256": dt.sha256_arquivo(caminho),
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
    dt.escrever_json(caminho_metadata, montar_metadata())

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


class SnapshotFalso:
    """Simula o snapshot do Evidently apenas para o contrato de extracao."""

    def __init__(self, metricas: list[dict]) -> None:
        self._metricas = metricas

    def dict(self) -> dict:
        return {"metrics": self._metricas}


def metrica_drift(coluna: str, valor: object, metodo: str = "K-S p_value") -> dict:
    return {
        "config": {
            "type": dt.TIPO_DRIFT,
            "column": coluna,
            "method": metodo,
            "threshold": 0.05,
        },
        "value": valor,
    }


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("method")
    caminho = tmp_path / "deteccao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(dt.ErroConfiguracao, match="Chaves ausentes"):
        dt.carregar_config(caminho)


def test_carregar_config_rejeita_etapas_iguais(tmp_path: Path) -> None:
    config = montar_config()
    config["batch"]["producer_stage"] = "treinamento"
    caminho = tmp_path / "deteccao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(dt.ErroConfiguracao, match="devem ser diferentes"):
        dt.carregar_config(caminho)


def test_carregar_config_rejeita_limiar_invalido(tmp_path: Path) -> None:
    config = montar_config()
    config["method"]["distance_threshold"] = 1.5
    caminho = tmp_path / "deteccao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(dt.ErroConfiguracao, match="entre 0 e 1"):
        dt.carregar_config(caminho)


def test_carregar_config_exige_ambos_limiares(tmp_path: Path) -> None:
    config = montar_config()
    config["method"].pop("p_value_threshold")
    caminho = tmp_path / "deteccao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(dt.ErroConfiguracao, match="p_value_threshold"):
        dt.carregar_config(caminho)


def test_carregar_manifesto_rejeita_status_nao_sucesso(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.json"
    caminho.write_text(
        json.dumps({"contract_version": "1.0", "status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(dt.ErroDeteccao, match="nao concluiu com sucesso"):
        dt.carregar_manifesto(caminho)


def test_separar_manifestos_rejeita_ausente() -> None:
    manifesto = {"stage": "treinamento", "run_id": "r1"}

    with pytest.raises(dt.ErroDeteccao, match="Manifestos ausentes"):
        dt.separar_manifestos([manifesto], montar_config())


def test_separar_manifestos_rejeita_duplicado() -> None:
    duplicados = [
        {"stage": "treinamento", "run_id": "r1"},
        {"stage": "treinamento", "run_id": "r2"},
        {"stage": "simulacao", "run_id": "r3"},
    ]

    with pytest.raises(dt.ErroDeteccao, match="duplicado"):
        dt.separar_manifestos(duplicados, montar_config())


def test_verificar_artefato_rejeita_hash_divergente(tmp_path: Path) -> None:
    caminho = tmp_path / "artefato.csv"
    caminho.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(dt.ErroDeteccao, match="Hash divergente"):
        dt.verificar_artefato({"path": caminho.as_posix(), "sha256": "0" * 64})


def test_ler_features_aplica_tipos(tmp_path: Path) -> None:
    caminho = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "record_id": ["LC1", "LC2"],
            "annual_inc": ["60000.5", ""],
            "term": ["36 months", "60 months"],
        }
    ).to_csv(caminho, index=False, encoding="utf-8")

    base = dt.ler_features(caminho, ["annual_inc"], ["term"])

    assert pd.api.types.is_numeric_dtype(base["annual_inc"])
    assert np.isnan(base.loc[1, "annual_inc"])
    assert base["term"].dtype == object


def test_avaliar_aplica_regra_por_metodo() -> None:
    assert dt.avaliar(0.01, "K-S p_value", 0.05, 0.1) == (0.05, True)
    assert dt.avaliar(0.20, "K-S p_value", 0.05, 0.1) == (0.05, False)
    assert dt.avaliar(0.90, "Wasserstein distance", 0.05, 0.1) == (0.1, True)
    assert dt.avaliar(0.05, "Jensen-Shannon distance", 0.05, 0.1) == (0.1, False)


def test_montar_resultados_marca_variavel_nao_avaliavel() -> None:
    snapshot = SnapshotFalso([metrica_drift("annual_inc", 0.001)])

    resultados = dt.montar_resultados(snapshot, ["annual_inc", "dti"], 0.05, 0.1)

    por_feature = {item["feature"]: item for item in resultados}
    assert por_feature["annual_inc"]["drift_detected"] is True
    assert por_feature["annual_inc"]["threshold"] == 0.05
    assert por_feature["dti"]["drift_detected"] is None
    assert por_feature["dti"]["statistic"] is None
    assert "reason" in por_feature["dti"]


def test_montar_resultados_marca_nulo_como_nao_avaliavel() -> None:
    snapshot = SnapshotFalso([metrica_drift("annual_inc", None)])

    resultados = dt.montar_resultados(snapshot, ["annual_inc"], 0.05, 0.1)

    assert resultados[0]["drift_detected"] is None
    assert "nao avaliavel" in resultados[0]["reason"]


def test_resumir_calcula_proporcao() -> None:
    resultados = [
        {"feature": "a", "drift_detected": True},
        {"feature": "b", "drift_detected": False},
        {"feature": "c", "drift_detected": None},
    ]

    resumo = dt.resumir(resultados)

    assert resumo["evaluated_columns"] == 2
    assert resumo["drifted_columns"] == 1
    assert resumo["drift_share"] == 0.5
    assert resumo["dataset_drift"] is False
    assert resumo["drifted_features"] == ["a"]


def test_processar_publica_drift_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "deteccao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-det"

    codigo = dt.processar(caminho_config, config, manifestos, "run-det", output_dir)

    assert codigo == dt.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "deteccao"
    assert manifesto["status"] == "success"
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["reference_id"] == "ref-teste"
    assert manifesto["scenario_id"] == "data-drift"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["schema_version"] == "1.0"

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == {"drift_json", "drift_html"}
    for item in saidas.values():
        assert Path(item["path"]).is_file()
        assert dt.sha256_arquivo(Path(item["path"])) == item["sha256"]

    documento = json.loads((output_dir / "drift.json").read_text(encoding="utf-8"))
    assert documento["tool"]["name"] == "evidently"
    assert documento["method"]["p_value_threshold"] == 0.05
    assert documento["method"]["distance_threshold"] == 0.1
    assert documento["samples"]["reference"]["row_count"] == QUANTIDADE
    assert documento["samples"]["current"]["row_count"] == QUANTIDADE
    assert {item["feature"] for item in documento["features"]} == set(FEATURES)
    assert documento["summary"]["dataset_drift"] is True
    assert set(documento["changed_variables"]["detected"]) == {"annual_inc", "dti", "revol_util"}

    for item in documento["features"]:
        assert item["drift_detected"] is True
        assert item["threshold"] in (0.05, 0.1)
        assert item["statistic"] is not None

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert nomes == {"variaveis_avaliadas", "variaveis_com_drift", "proporcao_drift"}


def test_processar_emite_alerta_de_drift(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "deteccao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-alerta"

    dt.processar(caminho_config, config, manifestos, "run-alerta", output_dir)

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "drift_alert", "completed"]
    alerta = next(evento for evento in eventos if evento["event"] == "drift_alert")
    assert alerta["level"] == "WARNING"


def test_processar_rejeita_dataset_divergente(tmp_path: Path) -> None:
    config = montar_config()
    config["dataset_id"] = "outro-dataset"
    caminho_config = tmp_path / "deteccao.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-det"

    codigo = dt.processar(caminho_config, config, manifestos, "run-det", output_dir)

    assert codigo == dt.CODIGO_ERRO
    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert eventos[-1]["event"] == "failed"
    assert eventos[-1]["error_code"] == "DETECCAO_INVALIDA"
