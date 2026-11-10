"""Configuracao do cenario de Concept Drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.manifestos import versao_maior

CONTRACT_VERSION = "1.0"
STAGE = "simulacao"
IDENTIFICADOR = "record_id"
CODIGO_SUCESSO = 0
CODIGO_ERRO = 1
OPERADORES = (">=", ">", "<=", "<")
OPERACAO = "raise_default_probability"


class ErroSimulacao(Exception):
    """Indica entrada invalida para a geracao do cenario de Concept Drift."""


def numero(valor: Any) -> bool:
    """Indica se o valor e um numero utilizavel (exclui booleanos)."""
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao do cenario de Concept Drift."""
    if not caminho.is_file():
        raise ErroSimulacao(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = (
        "dataset_id",
        "schema_version",
        "scenario_id",
        "base",
        "target",
        "concept",
        "outputs",
    )
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ValueError(f"Chaves ausentes na configuracao: {ausentes}")

    alvo = config["target"]
    for chave in ("column", "positive_value", "negative_value"):
        if chave not in alvo:
            raise ValueError(f"O alvo exige a chave '{chave}'.")

    conceito = config["concept"]
    if not conceito.get("rationale"):
        raise ValueError("O cenario de Concept Drift exige 'rationale'.")
    segmento = conceito.get("segment", {})
    if not segmento.get("column"):
        raise ValueError("O segmento exige a chave 'column'.")
    if segmento.get("operator") not in OPERADORES:
        raise ValueError(f"Operador de segmento invalido: {segmento.get('operator')}")
    if not numero(segmento.get("value")):
        raise ValueError("O segmento exige 'value' numerico.")

    perigo = conceito.get("hazard", {})
    for chave in ("reference_value", "slope", "scale"):
        if not numero(perigo.get(chave)):
            raise ValueError(f"O hazard exige '{chave}' numerico.")
    if not numero(perigo.get("max_probability")) or not 0 < float(perigo["max_probability"]) <= 1:
        raise ValueError("'max_probability' deve estar entre 0 (exclusivo) e 1.")
    if float(perigo["scale"]) <= 0:
        raise ValueError("'scale' deve ser positivo.")
    return config
