"""Testes da simulacao de Data Drift separados por responsabilidade."""

from __future__ import annotations

from pathlib import Path

import pytest

from monitoramento.simulacao.data_drift import config as config_mod

from .apoio import (
    configurar,
    escrever_config,
    executar,
    montar_config,
)


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("scenario_id")
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="Chaves ausentes"):
        config_mod.carregar_config(caminho)


def test_carregar_config_rejeita_menos_de_duas_variaveis(tmp_path: Path) -> None:
    config = montar_config()
    config["drift"]["variables"] = {
        "annual_inc": {"operation": "scale", "factor": 0.8, "clip": [0, 10000000]}
    }
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="ao menos 2 variaveis"):
        config_mod.carregar_config(caminho)


def test_carregar_config_rejeita_operacao_invalida(tmp_path: Path) -> None:
    config = montar_config()
    config["drift"]["variables"]["annual_inc"]["operation"] = "media"
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="Operacao invalida"):
        config_mod.carregar_config(caminho)


def test_carregar_config_rejeita_versao_incompativel(tmp_path: Path) -> None:
    config = montar_config()
    config["contract_version"] = "2.0"
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="incompativel"):
        config_mod.carregar_config(caminho)


@pytest.mark.parametrize("meses", [0, -1, 1.5, 24.0, "24", True])
def test_rejeita_deslocamento_sem_avanco_inteiro(tmp_path: Path, meses: object) -> None:
    config, parametros = configurar(tmp_path)
    config["time"]["shift_months"] = meses

    assert executar(config, parametros) == config_mod.CODIGO_ERRO
    assert not parametros["output_dir"].exists()
    assert not Path(config["outputs"]["analysis_dir"]).exists()
