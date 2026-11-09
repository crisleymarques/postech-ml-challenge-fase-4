"""Testes da simulacao de Concept Drift (issue F2-05)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoramento import simulacao_concept_drift as sc

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


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("concept")

    with pytest.raises(ValueError, match="Chaves ausentes"):
        sc.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_operador_invalido(tmp_path: Path) -> None:
    config = montar_config()
    config["concept"]["segment"]["operator"] = "=="

    with pytest.raises(ValueError, match="Operador"):
        sc.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_max_probability_invalida(tmp_path: Path) -> None:
    config = montar_config()
    config["concept"]["hazard"]["max_probability"] = 1.5

    with pytest.raises(ValueError, match="max_probability"):
        sc.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_scale_nao_positivo(tmp_path: Path) -> None:
    config = montar_config()
    config["concept"]["hazard"]["scale"] = 0

    with pytest.raises(ValueError, match="scale"):
        sc.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_rationale_ausente(tmp_path: Path) -> None:
    config = montar_config()
    config["concept"].pop("rationale")

    with pytest.raises(ValueError, match="rationale"):
        sc.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_arquivo_ausente(tmp_path: Path) -> None:
    with pytest.raises(sc.ErroSimulacao, match="nao encontrada"):
        sc.carregar_config(tmp_path / "inexistente.json")


def test_carregar_base_rejeita_arquivo_ausente(tmp_path: Path) -> None:
    with pytest.raises(sc.ErroSimulacao, match="nao encontrada"):
        sc.carregar_base(tmp_path / "inexistente.csv")


def test_mascara_segmento_aplica_operadores() -> None:
    valores = pd.Series([10.0, 20.0, 25.0, 30.0])

    assert sc.mascara_segmento(valores, {"operator": ">=", "value": 25.0}).tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert sc.mascara_segmento(valores, {"operator": ">", "value": 25.0}).tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert sc.mascara_segmento(valores, {"operator": "<=", "value": 20.0}).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert sc.mascara_segmento(valores, {"operator": "<", "value": 20.0}).tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_calcular_perigo_zera_fora_do_segmento_e_limita_maximo() -> None:
    valores = pd.Series([10.0, 20.0, 30.0, 60.0])
    mascara = np.array([False, False, True, True])
    perigo = {"reference_value": 25.0, "slope": 0.1, "scale": 1.0, "max_probability": 0.5}

    resultado = sc.calcular_perigo(valores, mascara, perigo)

    assert resultado[0] == 0.0
    assert resultado[1] == 0.0
    assert resultado[2] == pytest.approx(0.5)
    assert resultado[3] == pytest.approx(0.5)


def test_aplicar_mudanca_conceito_altera_apenas_negativos_do_segmento() -> None:
    base = gerar_base()
    alterada, descricao = sc.aplicar_mudanca_conceito(
        base, "loan_status", 1, montar_config()["concept"], SEMENTE
    )

    original = pd.to_numeric(base["loan_status"]).to_numpy()
    novo = pd.to_numeric(alterada["loan_status"]).to_numpy()
    dti = pd.to_numeric(base["dti"]).to_numpy()

    assert (novo[original == 1] == 1).all()
    fora = dti < 25
    assert (novo[fora] == original[fora]).all()

    virados = (original == 0) & (novo == 1)
    assert virados.sum() == descricao["flipped_rows"]
    assert int(virados.sum()) > 0
    assert (dti[virados] >= 25).all()
    assert descricao["positive_rate_after"] >= descricao["positive_rate_before"]
    assert descricao["operation"] == sc.OPERACAO
    assert descricao["segment_rows"] == int((dti >= 25).sum())


def test_aplicar_mudanca_conceito_reprodutivel() -> None:
    base = gerar_base()
    conceito = montar_config()["concept"]

    primeira, descricao_a = sc.aplicar_mudanca_conceito(base, "loan_status", 1, conceito, SEMENTE)
    segunda, descricao_b = sc.aplicar_mudanca_conceito(base, "loan_status", 1, conceito, SEMENTE)

    pd.testing.assert_frame_equal(primeira, segunda)
    assert descricao_a["flipped_rows"] == descricao_b["flipped_rows"]


def test_aplicar_mudanca_conceito_rejeita_coluna_ausente() -> None:
    base = gerar_base()
    conceito = montar_config()["concept"]
    conceito["segment"]["column"] = "coluna_inexistente"

    with pytest.raises(sc.ErroSimulacao, match="Coluna do segmento ausente"):
        sc.aplicar_mudanca_conceito(base, "loan_status", 1, conceito, SEMENTE)


def test_separar_features_rotulos_mantem_identificador() -> None:
    base = gerar_base(5)

    features, rotulos = sc.separar_features_rotulos(base, "loan_status")

    assert "loan_status" not in features.columns
    assert list(rotulos.columns) == ["record_id", "loan_status"]
    assert len(features) == len(rotulos) == 5


def test_montar_medidas_descreve_cenario() -> None:
    descricao = {
        "segment_rows": 10,
        "flipped_rows": 4,
        "positive_rate_before": 0.2,
        "positive_rate_after": 0.3,
    }

    medidas = sc.montar_medidas(descricao, 100)
    nomes = {item["name"] for item in medidas}

    assert nomes == {
        "linhas_geradas",
        "registros_no_segmento",
        "rotulos_alterados",
        "taxa_positivos_antes",
        "taxa_positivos_depois",
    }


def test_gerar_publica_artefatos_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    origem = tmp_path / "simulation_base.csv"
    gerar_base().to_csv(origem, index=False, encoding="utf-8")
    config["base"]["path"] = origem.as_posix()
    caminho_config = escrever_config(tmp_path, config)
    output_dir = tmp_path / "runs" / "run-concept"
    analysis_dir = tmp_path / "analysis"

    codigo = sc.gerar(
        config_path=caminho_config,
        source_path=None,
        analysis_dir=analysis_dir,
        batch_id="batch-1",
        run_id="run-concept",
        output_dir=output_dir,
        seed=SEMENTE,
    )

    assert codigo == sc.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "simulacao"
    assert manifesto["status"] == "success"
    assert manifesto["scenario_id"] == "concept-drift"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["drift_type"] == "concept"
    assert manifesto["features_unchanged"] is True
    assert manifesto["concept_change"]["flipped_rows"] > 0

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == {"features", "labels", "labels_reference"}
    for item in saidas.values():
        assert Path(item["path"]).is_file()
        assert sc.sha256_arquivo(Path(item["path"])) == item["sha256"]

    features = pd.read_csv(saidas["features"]["path"])
    assert "loan_status" not in features.columns
    rotulos = pd.read_csv(saidas["labels"]["path"])
    assert list(rotulos.columns) == ["record_id", "loan_status"]
    referencia = pd.read_csv(saidas["labels_reference"]["path"])
    assert rotulos["loan_status"].sum() > referencia["loan_status"].sum()

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "completed"]

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metricas["run_id"] == "run-concept"
    assert any(item["name"] == "rotulos_alterados" for item in metricas["measurements"])
