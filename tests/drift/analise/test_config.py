"""Testes de config da analise estatistica."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.drift.analise import config as configuracao

from .apoio import (
    montar_config,
)


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("method")
    caminho = tmp_path / "analise.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="Chaves ausentes"):
        configuracao.carregar_config(caminho)


def test_carregar_config_rejeita_etapas_iguais(tmp_path: Path) -> None:
    config = montar_config()
    config["batch"]["producer_stage"] = "treinamento"
    caminho = tmp_path / "analise.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="devem ser diferentes"):
        configuracao.carregar_config(caminho)


def test_carregar_config_rejeita_limiar_invalido(tmp_path: Path) -> None:
    config = montar_config()
    config["method"]["psi_threshold"] = 1.5
    caminho = tmp_path / "analise.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="entre 0 e 1"):
        configuracao.carregar_config(caminho)


def test_carregar_config_rejeita_bins_invalido(tmp_path: Path) -> None:
    config = montar_config()
    config["method"]["bins"] = 1
    caminho = tmp_path / "analise.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(configuracao.ErroConfiguracao, match="bins"):
        configuracao.carregar_config(caminho)
