"""Testes de config dos alertas."""

from __future__ import annotations

from pathlib import Path

import pytest

from monitoramento.observabilidade.alertas import config as al

from .apoio import CONFIG_REAL, config_minimo, escrever_config


def test_carregar_config_real() -> None:
    config = al.carregar_config(CONFIG_REAL)
    assert config["severidade_por_situacao"]["critico"] == "critical"
    assert "critical" in config["severidades"]


def test_carregar_config_ausente(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nao encontrada"):
        al.carregar_config(tmp_path / "inexistente.json")


def test_carregar_config_versao_incompativel(tmp_path: Path) -> None:
    caminho = escrever_config(tmp_path, config_minimo(contract_version="2.0"))
    with pytest.raises(ValueError, match="incompativel"):
        al.carregar_config(caminho)


def test_carregar_config_sem_severidades(tmp_path: Path) -> None:
    caminho = escrever_config(tmp_path, config_minimo(severidades=[]))
    with pytest.raises(ValueError, match="severidades"):
        al.carregar_config(caminho)


def test_carregar_config_sem_critico(tmp_path: Path) -> None:
    caminho = escrever_config(tmp_path, config_minimo(severidades=["warning"]))
    with pytest.raises(ValueError, match="critical"):
        al.carregar_config(caminho)


def test_carregar_config_mapa_invalido(tmp_path: Path) -> None:
    caminho = escrever_config(
        tmp_path, config_minimo(severidade_por_situacao={"atencao": "warning"})
    )
    with pytest.raises(ValueError, match="severidade_por_situacao"):
        al.carregar_config(caminho)


def test_carregar_config_severidade_falha_invalida(tmp_path: Path) -> None:
    caminho = escrever_config(tmp_path, config_minimo(severidade_falha="urgente"))
    with pytest.raises(ValueError, match="severidade_falha"):
        al.carregar_config(caminho)
