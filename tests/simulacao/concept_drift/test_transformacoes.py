"""Testes de transformacoes de Concept Drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoramento.simulacao.concept_drift import config as configuracao
from monitoramento.simulacao.concept_drift import transformacoes

from .apoio import SEMENTE, gerar_base, montar_config


def test_mascara_segmento_aplica_operadores() -> None:
    valores = pd.Series([10.0, 20.0, 25.0, 30.0])

    assert transformacoes.mascara_segmento(valores, {"operator": ">=", "value": 25.0}).tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert transformacoes.mascara_segmento(valores, {"operator": ">", "value": 25.0}).tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert transformacoes.mascara_segmento(valores, {"operator": "<=", "value": 20.0}).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert transformacoes.mascara_segmento(valores, {"operator": "<", "value": 20.0}).tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_calcular_perigo_zera_fora_do_segmento_e_limita_maximo() -> None:
    valores = pd.Series([10.0, 20.0, 30.0, 60.0])
    mascara = np.array([False, False, True, True])
    perigo = {"reference_value": 25.0, "slope": 0.1, "scale": 1.0, "max_probability": 0.5}

    resultado = transformacoes.calcular_perigo(valores, mascara, perigo)

    assert resultado[0] == 0.0
    assert resultado[1] == 0.0
    assert resultado[2] == pytest.approx(0.5)
    assert resultado[3] == pytest.approx(0.5)


def test_aplicar_mudanca_conceito_altera_apenas_negativos_do_segmento() -> None:
    base = gerar_base()
    alterada, descricao = transformacoes.aplicar_mudanca_conceito(
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
    assert descricao["operation"] == configuracao.OPERACAO
    assert descricao["segment_rows"] == int((dti >= 25).sum())


def test_aplicar_mudanca_conceito_reprodutivel() -> None:
    base = gerar_base()
    conceito = montar_config()["concept"]

    primeira, descricao_a = transformacoes.aplicar_mudanca_conceito(
        base, "loan_status", 1, conceito, SEMENTE
    )
    segunda, descricao_b = transformacoes.aplicar_mudanca_conceito(
        base, "loan_status", 1, conceito, SEMENTE
    )

    pd.testing.assert_frame_equal(primeira, segunda)
    assert descricao_a["flipped_rows"] == descricao_b["flipped_rows"]


def test_aplicar_mudanca_conceito_rejeita_coluna_ausente() -> None:
    base = gerar_base()
    conceito = montar_config()["concept"]
    conceito["segment"]["column"] = "coluna_inexistente"

    with pytest.raises(configuracao.ErroSimulacao, match="Coluna do segmento ausente"):
        transformacoes.aplicar_mudanca_conceito(base, "loan_status", 1, conceito, SEMENTE)


def test_separar_features_rotulos_mantem_identificador() -> None:
    base = gerar_base(5)

    features, rotulos = transformacoes.separar_features_rotulos(base, "loan_status")

    assert "loan_status" not in features.columns
    assert list(rotulos.columns) == ["record_id", "loan_status"]
    assert len(features) == len(rotulos) == 5
