"""Testes de entradas da analise estatistica."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoramento.drift import dados
from monitoramento.drift.analise import config as configuracao
from monitoramento.drift.analise import entradas

from .apoio import (
    montar_config,
)


def test_carregar_manifesto_rejeita_status_nao_sucesso(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.json"
    caminho.write_text(
        json.dumps({"contract_version": "1.0", "status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(configuracao.ErroAnalise, match="nao concluiu com sucesso"):
        entradas.carregar_manifesto(caminho)


def test_separar_manifestos_rejeita_ausente() -> None:
    manifesto = {"stage": "treinamento", "run_id": "r1"}

    with pytest.raises(configuracao.ErroAnalise, match="Manifestos ausentes"):
        entradas.separar_manifestos([manifesto], montar_config())


def test_separar_manifestos_rejeita_duplicado() -> None:
    duplicados = [
        {"stage": "treinamento", "run_id": "r1"},
        {"stage": "treinamento", "run_id": "r2"},
        {"stage": "simulacao", "run_id": "r3"},
    ]

    with pytest.raises(configuracao.ErroAnalise, match="duplicado"):
        entradas.separar_manifestos(duplicados, montar_config())


def test_verificar_artefato_rejeita_hash_divergente(tmp_path: Path) -> None:
    caminho = tmp_path / "artefato.csv"
    caminho.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(configuracao.ErroAnalise, match="Hash divergente"):
        entradas.verificar_artefato({"path": caminho.as_posix(), "sha256": "0" * 64})


def test_ler_features_aplica_tipos(tmp_path: Path) -> None:
    caminho = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "record_id": ["LC1", "LC2"],
            "annual_inc": ["60000.5", ""],
            "term": ["36 months", "60 months"],
        }
    ).to_csv(caminho, index=False, encoding="utf-8")

    base = dados.ler_features(caminho, ["annual_inc"], ["term"])

    assert pd.api.types.is_numeric_dtype(base["annual_inc"])
    assert np.isnan(base.loc[1, "annual_inc"])
    assert base["term"].dtype == object
