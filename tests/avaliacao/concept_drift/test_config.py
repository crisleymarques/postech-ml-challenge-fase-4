"""Testes de config de Concept Drift."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.avaliacao.concept_drift import (
    config as configuracao,
)

from .apoio import (
    montar_config,
)


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("metrics")
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="Chaves ausentes"):
        configuracao.carregar_config(caminho)


def test_carregar_config_rejeita_etapas_iguais(tmp_path: Path) -> None:
    config = montar_config()
    config["batch"]["producer_stage"] = "treinamento"
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="devem ser diferentes"):
        configuracao.carregar_config(caminho)


def test_carregar_config_rejeita_metrica_invalida(tmp_path: Path) -> None:
    config = montar_config()
    config["metrics"] = ["accuracy", "xpto"]
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="invalidas"):
        configuracao.carregar_config(caminho)


def test_carregar_config_rejeita_metrica_de_degradacao_fora_da_lista(tmp_path: Path) -> None:
    config = montar_config()
    config["metrics"] = ["accuracy"]
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="degradacao"):
        configuracao.carregar_config(caminho)


def test_carregar_config_rejeita_min_delta_nao_positivo(tmp_path: Path) -> None:
    config = montar_config()
    config["degradation"]["min_delta"] = 0
    caminho = tmp_path / "avaliacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="min_delta"):
        configuracao.carregar_config(caminho)


def test_carregar_config_rejeita_arquivo_ausente(tmp_path: Path) -> None:
    with pytest.raises(configuracao.ErroConfiguracao, match="nao encontrada"):
        configuracao.carregar_config(tmp_path / "inexistente.json")
