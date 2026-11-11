"""Configuracao da analise estatistica de Data Drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.manifestos import versao_maior

CONTRACT_VERSION = "1.0"


STAGE = "analise"


IDENTIFICADOR = "record_id"


CODIGO_SUCESSO = 0


CODIGO_ERRO = 1


METODO = "psi_ks"


LIMIARES_CONFIG = (
    "psi_moderate",
    "psi_threshold",
    "ks_p_value_threshold",
    "ks_statistic_threshold",
)


TIPO_NUMERICO = "numeric"


TIPO_CATEGORICO = "categorical"


CRITERIO_PSI = (
    "PSI >= 'psi_threshold' indica mudanca significativa; PSI entre "
    "'psi_moderate' e 'psi_threshold' indica mudanca moderada; abaixo disso, "
    "insignificante."
)


CRITERIO_KS = (
    "K-S (variaveis numericas) indica mudanca quando o p-valor fica abaixo de "
    "'ks_p_value_threshold' e a estatistica atinge 'ks_statistic_threshold'."
)


class ErroAnalise(Exception):
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
    """Valida a configuracao antes de reservar uma execucao."""
    if not isinstance(config, dict):
        raise ErroConfiguracao("Configuracao deve ser um objeto JSON.")
    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroConfiguracao("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "reference", "batch", "method")
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ErroConfiguracao(f"Chaves ausentes na configuracao: {ausentes}")

    for papel in ("reference", "batch"):
        if not isinstance(config[papel], dict) or not config[papel].get("producer_stage"):
            raise ErroConfiguracao(f"O papel '{papel}' exige a chave 'producer_stage'.")
    if config["reference"]["producer_stage"] == config["batch"]["producer_stage"]:
        raise ErroConfiguracao("As etapas produtoras da referencia e do lote devem ser diferentes.")

    if not isinstance(config["method"], dict) or not config["method"].get("name"):
        raise ErroConfiguracao("O metodo de analise exige a chave 'name'.")
    bins = config["method"].get("bins")
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ErroConfiguracao("'bins' deve ser um inteiro maior ou igual a 2.")
    for chave in LIMIARES_CONFIG:
        limiar = config["method"].get(chave)
        if not isinstance(limiar, (int, float)) or not 0 < float(limiar) < 1:
            raise ErroConfiguracao(f"'{chave}' deve ser um numero entre 0 e 1.")
