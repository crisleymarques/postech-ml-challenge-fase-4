"""Testes de interpretacao da analise estatistica."""
from __future__ import annotations

import numpy as np
import pandas as pd

from monitoramento.drift.analise import config as configuracao
from monitoramento.drift.analise import interpretacao

from .apoio import (
    FEATURES,
    NUMERICAS,
    QUANTIDADE,
    SEMENTE,
    gerar_features,
)


def test_analisar_variavel_numerica_detecta_mudanca() -> None:
    rng = np.random.default_rng(SEMENTE)
    referencia = pd.Series(rng.normal(60000, 5000, QUANTIDADE))
    atual = pd.Series(rng.normal(120000, 5000, QUANTIDADE))

    resultado = interpretacao.analisar_variavel(
        "annual_inc",
        configuracao.TIPO_NUMERICO,
        referencia,
        atual,
        bins=10,
        psi_moderado=0.1,
        psi_limiar=0.25,
        ks_p=0.05,
        ks_limiar=0.1,
    )

    assert resultado["drift_detected"] is True
    assert resultado["psi_band"] == "significativa"
    assert resultado["ks_statistic"] is not None
    assert resultado["reasons"]


def test_analisar_variavel_nao_avaliavel_registra_motivo() -> None:
    vazia = pd.Series([None, None], dtype="float64")

    resultado = interpretacao.analisar_variavel(
        "annual_inc",
        configuracao.TIPO_NUMERICO,
        vazia,
        vazia,
        bins=10,
        psi_moderado=0.1,
        psi_limiar=0.25,
        ks_p=0.05,
        ks_limiar=0.1,
    )

    assert resultado["drift_detected"] is None
    assert resultado["psi"] is None
    assert resultado["reasons"]


def test_montar_resultados_preserva_ordem() -> None:
    referencia = gerar_features(deslocado=False)
    atual = gerar_features(deslocado=True)

    resultados = interpretacao.montar_resultados(
        referencia,
        atual,
        FEATURES,
        NUMERICAS,
        bins=10,
        psi_moderado=0.1,
        psi_limiar=0.25,
        ks_p=0.05,
        ks_limiar=0.1,
    )

    assert [item["feature"] for item in resultados] == FEATURES
    assert all(item["drift_detected"] is True for item in resultados)


def test_resumir_calcula_proporcao_e_psi_maximo() -> None:
    resultados = [
        {"feature": "a", "drift_detected": True, "psi_band": "significativa", "psi": 0.4},
        {"feature": "b", "drift_detected": False, "psi_band": "insignificante", "psi": 0.02},
        {"feature": "c", "drift_detected": None, "psi_band": None, "psi": None},
    ]

    resumo = interpretacao.resumir(resultados)

    assert resumo["evaluated_columns"] == 2
    assert resumo["changed_columns"] == 1
    assert resumo["change_share"] == 0.5
    assert resumo["significant_columns"] == 1
    assert resumo["psi_maximo"] == 0.4
    assert resumo["changed_features"] == ["a"]


def test_montar_conclusoes_registra_confirmadas_e_ressalva() -> None:
    resumo = {
        "evaluated_columns": 3,
        "changed_columns": 2,
        "significant_columns": 2,
        "changed_features": ["annual_inc", "dti"],
    }

    conclusoes = interpretacao.montar_conclusoes(
        resumo, ["annual_inc", "dti", "revol_util"], psi_limiar=0.25, ks_p=0.05
    )

    texto = " ".join(conclusoes)
    assert "revol_util" in texto
    assert "nao comprova" in texto
