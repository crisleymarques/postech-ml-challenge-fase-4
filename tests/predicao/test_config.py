"""Testes migrados da etapa de predicao."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.predicao import config as cfg

from .apoio import (
    montar_config,
)


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("batch")
    caminho = tmp_path / "predicao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(cfg.ErroConfiguracao, match="Chaves ausentes"):
        cfg.carregar_config(caminho)


def test_carregar_config_rejeita_etapas_iguais(tmp_path: Path) -> None:
    config = montar_config()
    config["batch"]["producer_stage"] = "treinamento"
    caminho = tmp_path / "predicao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(cfg.ErroConfiguracao, match="devem ser diferentes"):
        cfg.carregar_config(caminho)
