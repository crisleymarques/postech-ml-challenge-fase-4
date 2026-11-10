"""Configuracao da avaliacao de Concept Drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.manifestos import versao_maior

CONTRACT_VERSION = "1.0"
STAGE = "avaliacao"
IDENTIFICADOR = "record_id"
CODIGO_SUCESSO = 0
CODIGO_ERRO = 1
METRICAS = ("accuracy", "precision", "recall", "f1", "roc_auc")


class ErroAvaliacao(Exception):
    """Indica dependencia ausente, divergente ou incompativel."""


class ErroConfiguracao(Exception):
    """Indica configuracao ausente ou invalida."""


def numero(valor: Any) -> bool:
    """Indica se o valor e um numero utilizavel (exclui booleanos)."""
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao da avaliacao."""
    if not caminho.is_file():
        raise ErroConfiguracao(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroConfiguracao("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "model", "batch", "metrics")
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ErroConfiguracao(f"Chaves ausentes na configuracao: {ausentes}")

    for papel in ("model", "batch"):
        if not config[papel].get("producer_stage"):
            raise ErroConfiguracao(f"O papel '{papel}' exige a chave 'producer_stage'.")
    if config["model"]["producer_stage"] == config["batch"]["producer_stage"]:
        raise ErroConfiguracao("As etapas produtoras do modelo e do lote devem ser diferentes.")

    invalidas = [metrica for metrica in config["metrics"] if metrica not in METRICAS]
    if invalidas:
        raise ErroConfiguracao(f"Metricas invalidas: {invalidas}. Esperado: {list(METRICAS)}.")

    degradacao = config.get("degradation", {})
    if degradacao:
        if degradacao.get("metric") not in config["metrics"]:
            raise ErroConfiguracao("A metrica de degradacao deve constar em 'metrics'.")
        if not numero(degradacao.get("min_delta")) or float(degradacao["min_delta"]) <= 0:
            raise ErroConfiguracao("'min_delta' deve ser um numero positivo.")
    return config
