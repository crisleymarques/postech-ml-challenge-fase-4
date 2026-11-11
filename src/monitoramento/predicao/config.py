"""Configuracao da etapa de predicao."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.manifestos import versao_maior

CONTRACT_VERSION = "1.0"
STAGE = "predicao"
IDENTIFICADOR = "record_id"
CODIGO_SUCESSO = 0
CODIGO_ERRO = 1


class ErroPredicao(Exception):
    """Indica dependencia ausente, divergente ou incompativel."""


class ErroConfiguracao(Exception):
    """Indica configuracao ausente ou invalida."""


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao da etapa."""
    if not caminho.is_file():
        raise ErroConfiguracao(f"Configuracao nao encontrada: {caminho}")
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            config = json.load(arquivo)
    except (OSError, ValueError) as erro:
        raise ErroConfiguracao("Nao foi possivel ler a configuracao JSON.") from erro
    validar_config(config)
    return config


def validar_config(config: dict[str, Any]) -> None:
    """Valida a configuracao antes de iniciar uma execucao."""
    if not isinstance(config, dict):
        raise ErroConfiguracao("A configuracao deve ser um objeto JSON.")

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroConfiguracao("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "model", "batch", "validation", "outputs")
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ErroConfiguracao(f"Chaves ausentes na configuracao: {ausentes}")

    for papel in ("model", "batch", "validation"):
        if not isinstance(config[papel], dict) or not config[papel].get("producer_stage"):
            raise ErroConfiguracao(f"O papel '{papel}' exige a chave 'producer_stage'.")
    etapas = {"model": "treinamento", "batch": "simulacao", "validation": "validacao"}
    produtoras = [config[papel]["producer_stage"] for papel in etapas]
    if len(set(produtoras)) != 3:
        raise ErroConfiguracao("As etapas produtoras devem ser diferentes.")
    if any(config[papel]["producer_stage"] != etapa for papel, etapa in etapas.items()):
        raise ErroConfiguracao("Etapas esperadas: treinamento, simulacao e validacao.")
    obrigatorias = {
        "model": ("pipeline_artifact", "metadata_artifact"),
        "batch": ("features_artifact",),
        "outputs": ("analysis_dir",),
    }
    for papel, campos in obrigatorias.items():
        if not isinstance(config[papel], dict) or any(
            not isinstance(config[papel].get(campo), str) or not config[papel][campo].strip()
            for campo in campos
        ):
            raise ErroConfiguracao(f"Configuracao incompleta para '{papel}'.")
