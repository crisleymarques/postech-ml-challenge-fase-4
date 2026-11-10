"""Configuracao da etapa de deteccao de Data Drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.manifestos import versao_maior

CONTRACT_VERSION = "1.0"
STAGE = "deteccao"
IDENTIFICADOR = "record_id"
CODIGO_SUCESSO = 0
CODIGO_ERRO = 1
PRESET = "DataDriftPreset"
LIMIARES_CONFIG = ("p_value_threshold", "distance_threshold")


class ErroDeteccao(Exception):
    """Indica dependencia ausente, divergente ou incompativel."""


class ErroConfiguracao(Exception):
    """Indica configuracao ausente ou invalida."""


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao da etapa."""
    if not caminho.is_file():
        raise ErroConfiguracao(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroConfiguracao("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "reference", "batch", "method")
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ErroConfiguracao(f"Chaves ausentes na configuracao: {ausentes}")

    for papel in ("reference", "batch"):
        if not config[papel].get("producer_stage"):
            raise ErroConfiguracao(f"O papel '{papel}' exige a chave 'producer_stage'.")
    if config["reference"]["producer_stage"] == config["batch"]["producer_stage"]:
        raise ErroConfiguracao("As etapas produtoras da referencia e do lote devem ser diferentes.")

    if not config["method"].get("name"):
        raise ErroConfiguracao("O metodo de deteccao exige a chave 'name'.")
    for chave in LIMIARES_CONFIG:
        limiar = config["method"].get(chave)
        if not isinstance(limiar, (int, float)) or not 0 < float(limiar) < 1:
            raise ErroConfiguracao(f"'{chave}' deve ser um numero entre 0 e 1.")
    return config
