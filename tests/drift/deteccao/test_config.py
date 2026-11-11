"""Testes da deteccao de Data Drift por responsabilidade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.drift.deteccao import config as config_dt

from .apoio import montar_config


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("method")
    caminho = tmp_path / "deteccao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(config_dt.ErroConfiguracao, match="Chaves ausentes"):
        config_dt.carregar_config(caminho)


def test_carregar_config_rejeita_etapas_iguais(tmp_path: Path) -> None:
    config = montar_config()
    config["batch"]["producer_stage"] = "treinamento"
    caminho = tmp_path / "deteccao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(config_dt.ErroConfiguracao, match="devem ser diferentes"):
        config_dt.carregar_config(caminho)


def test_carregar_config_rejeita_limiar_invalido(tmp_path: Path) -> None:
    config = montar_config()
    config["method"]["distance_threshold"] = 1.5
    caminho = tmp_path / "deteccao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(config_dt.ErroConfiguracao, match="entre 0 e 1"):
        config_dt.carregar_config(caminho)


def test_carregar_config_exige_ambos_limiares(tmp_path: Path) -> None:
    config = montar_config()
    config["method"].pop("p_value_threshold")
    caminho = tmp_path / "deteccao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(config_dt.ErroConfiguracao, match="p_value_threshold"):
        config_dt.carregar_config(caminho)
