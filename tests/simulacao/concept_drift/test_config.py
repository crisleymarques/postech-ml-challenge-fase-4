"""Testes de config de Concept Drift."""
from __future__ import annotations

from pathlib import Path

import pytest

from monitoramento.simulacao.concept_drift import config as configuracao

from .apoio import escrever_config, montar_config


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("concept")

    with pytest.raises(ValueError, match="Chaves ausentes"):
        configuracao.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_operador_invalido(tmp_path: Path) -> None:
    config = montar_config()
    config["concept"]["segment"]["operator"] = "=="

    with pytest.raises(ValueError, match="Operador"):
        configuracao.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_max_probability_invalida(tmp_path: Path) -> None:
    config = montar_config()
    config["concept"]["hazard"]["max_probability"] = 1.5

    with pytest.raises(ValueError, match="max_probability"):
        configuracao.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_scale_nao_positivo(tmp_path: Path) -> None:
    config = montar_config()
    config["concept"]["hazard"]["scale"] = 0

    with pytest.raises(ValueError, match="scale"):
        configuracao.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_rationale_ausente(tmp_path: Path) -> None:
    config = montar_config()
    config["concept"].pop("rationale")

    with pytest.raises(ValueError, match="rationale"):
        configuracao.carregar_config(escrever_config(tmp_path, config))


def test_carregar_config_rejeita_arquivo_ausente(tmp_path: Path) -> None:
    with pytest.raises(configuracao.ErroSimulacao, match="nao encontrada"):
        configuracao.carregar_config(tmp_path / "inexistente.json")
