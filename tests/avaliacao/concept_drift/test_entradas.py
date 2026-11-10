"""Testes de entradas de Concept Drift."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento.avaliacao.concept_drift import (
    config as configuracao,
)
from monitoramento.avaliacao.concept_drift import (
    entradas,
)

from .apoio import (
    montar_config,
)


def test_carregar_manifesto_rejeita_status_nao_sucesso(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.json"
    caminho.write_text(
        json.dumps({"contract_version": "1.0", "status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(configuracao.ErroAvaliacao, match="nao concluiu com sucesso"):
        entradas.carregar_manifesto(caminho)


def test_separar_manifestos_rejeita_ausente() -> None:
    manifesto = {"stage": "treinamento", "run_id": "r1"}

    with pytest.raises(configuracao.ErroAvaliacao, match="Manifestos ausentes"):
        entradas.separar_manifestos([manifesto], montar_config())


def test_separar_manifestos_rejeita_duplicado() -> None:
    duplicados = [
        {"stage": "treinamento", "run_id": "r1"},
        {"stage": "treinamento", "run_id": "r2"},
        {"stage": "simulacao", "run_id": "r3"},
    ]

    with pytest.raises(configuracao.ErroAvaliacao, match="duplicado"):
        entradas.separar_manifestos(duplicados, montar_config())


def test_artefato_rejeita_tipo_ausente() -> None:
    with pytest.raises(configuracao.ErroAvaliacao, match="ausente"):
        entradas.artefato({"run_id": "r", "outputs": []}, "model_pipeline")


def test_verificar_artefato_rejeita_hash_divergente(tmp_path: Path) -> None:
    caminho = tmp_path / "artefato.csv"
    caminho.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(configuracao.ErroAvaliacao, match="Hash divergente"):
        entradas.verificar_artefato({"path": caminho.as_posix(), "sha256": "0" * 64})


def test_verificar_compatibilidade_rejeita_dataset_divergente() -> None:
    with pytest.raises(configuracao.ErroAvaliacao, match="dataset_id divergente"):
        entradas.verificar_compatibilidade(
            montar_config(),
            {"run_id": "r1", "dataset_id": "outro"},
            {"run_id": "r2", "dataset_id": "teste"},
        )


def test_ler_lote_trata_campo_vazio_como_ausencia(tmp_path: Path) -> None:
    caminho = tmp_path / "features.csv"
    pd.DataFrame({"record_id": ["LC1", "LC2"], "dti": ["10", ""]}).to_csv(
        caminho, index=False, encoding="utf-8"
    )

    lote = entradas.ler_lote(caminho)

    assert lote.loc[0, "dti"] == "10"
    assert pd.isna(lote.loc[1, "dti"])


def test_juntar_rotulos_alinhado_pelo_identificador(tmp_path: Path) -> None:
    caminho = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": ["LC2", "LC1"], "loan_status": ["1", "0"]}).to_csv(
        caminho, index=False, encoding="utf-8"
    )

    rotulos = entradas.juntar_rotulos(pd.Series(["LC1", "LC2"]), caminho, "loan_status")

    assert rotulos.tolist() == [0, 1]


def test_juntar_rotulos_rejeita_identificadores_repetidos(tmp_path: Path) -> None:
    caminho = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": ["LC1"], "loan_status": ["0"]}).to_csv(
        caminho, index=False, encoding="utf-8"
    )

    with pytest.raises(configuracao.ErroAvaliacao, match="repetidos"):
        entradas.juntar_rotulos(pd.Series(["LC1", "LC1"]), caminho, "loan_status")


def test_juntar_rotulos_rejeita_rotulos_ausentes(tmp_path: Path) -> None:
    caminho = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": ["LC1"], "loan_status": ["0"]}).to_csv(
        caminho, index=False, encoding="utf-8"
    )

    with pytest.raises(configuracao.ErroAvaliacao, match="Rotulos ausentes"):
        entradas.juntar_rotulos(pd.Series(["LC1", "LC2"]), caminho, "loan_status")
