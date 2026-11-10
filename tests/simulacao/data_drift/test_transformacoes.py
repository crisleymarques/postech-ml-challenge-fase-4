"""Testes da simulacao de Data Drift separados por responsabilidade."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoramento.simulacao.data_drift import config as config_mod
from monitoramento.simulacao.data_drift import transformacoes

from .apoio import (
    gerar_origem,
    montar_config,
)


def test_deslocar_periodo_avanca_meses() -> None:
    serie = pd.Series(["Jul-2017", "Dec-2017"])

    deslocada = transformacoes.deslocar_periodo(serie, "%b-%Y", 24)

    assert list(deslocada) == ["Jul-2019", "Dec-2019"]


def test_aplicar_drift_altera_media_e_respeita_limites() -> None:
    base = gerar_origem()
    variaveis = montar_config()["drift"]["variables"]

    ajustado, descritas = transformacoes.aplicar_drift(base, variaveis, seed=42, jitter=0.05)

    assert [item["column"] for item in descritas] == ["annual_inc", "dti", "revol_util"]
    assert ajustado["annual_inc"].mean() < base["annual_inc"].astype(float).mean() * 0.9
    assert ajustado["dti"].mean() > base["dti"].astype(float).mean()
    assert ajustado["revol_util"].max() == pytest.approx(1000.0)
    assert ajustado["record_id"].equals(base["record_id"])


def test_aplicar_drift_preserva_ausentes() -> None:
    base = gerar_origem()
    base.loc[3, "revol_util"] = ""

    ajustado, descritas = transformacoes.aplicar_drift(
        base, montar_config()["drift"]["variables"], 42, 0.05
    )

    assert ajustado.at[3, "revol_util"] == ""
    descricao = next(item for item in descritas if item["column"] == "revol_util")
    assert descricao["missing_preserved"] == 1


def test_aplicar_drift_rejeita_coluna_ausente() -> None:
    base = pd.DataFrame({"record_id": ["LC0000001"]})

    with pytest.raises(config_mod.ErroSimulacao, match="Coluna da alteracao ausente"):
        transformacoes.aplicar_drift(
            base, {"annual_inc": {"operation": "shift", "delta": 1.0, "clip": [0, 10]}}, 1, 0.0
        )


def test_separar_features_rotulos_remove_alvo() -> None:
    base = gerar_origem()

    features, rotulos = transformacoes.separar_features_rotulos(base, "loan_status")

    assert "loan_status" not in features.columns
    assert "record_id" in features.columns
    assert list(rotulos.columns) == ["record_id", "loan_status"]
    assert len(features) == len(rotulos) == len(base)


@pytest.mark.parametrize("ausente", ["", None, np.nan, pd.NA])
def test_aplicar_drift_preserva_apenas_ausencias_originais(base, config, ausente) -> None:
    base.loc[0, "dti"] = ausente

    ajustada, descritas = transformacoes.aplicar_drift(base, config["drift"]["variables"], 42, 0.05)

    assert ajustada.at[0, "dti"] == ""
    assert ajustada.loc[1:, "dti"].astype(float).gt(base.loc[1:, "dti"].astype(float)).all()
    descricao = next(item for item in descritas if item["column"] == "dti")
    assert descricao["missing_preserved"] == 1
