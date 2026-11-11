"""Configuracao e constantes da simulacao de Data Drift."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "1.0"
STAGE = "simulacao"
IDENTIFICADOR = "record_id"
CODIGO_SUCESSO = 0
CODIGO_ERRO = 1
OPERACOES = ("scale", "shift")
MINIMO_VARIAVEIS = 2


class ErroSimulacao(Exception):
    """Indica entrada invalida para a geracao do cenario simulado."""


def versao_maior(versao: Any) -> str:
    """Extrai o componente principal de uma versao de contrato."""
    return str(versao or "").split(".")[0]


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao da simulacao."""
    if not caminho.is_file():
        raise ErroSimulacao(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if not isinstance(config, dict):
        raise ValueError("A configuracao deve ser um objeto JSON.")
    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = (
        "dataset_id",
        "schema_version",
        "scenario_id",
        "base",
        "time",
        "target",
        "drift",
        "outputs",
    )
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ValueError(f"Chaves ausentes na configuracao: {ausentes}")

    campos = {
        "base": ("path",),
        "time": ("column", "format", "shift_months"),
        "target": ("column",),
        "drift": ("seed", "variables"),
        "outputs": ("analysis_dir",),
    }
    for secao, chaves in campos.items():
        if not isinstance(config[secao], dict) or any(
            chave not in config[secao] for chave in chaves
        ):
            raise ValueError(f"Configuracao incompleta na secao {secao}.")
    meses = config["time"]["shift_months"]
    if type(meses) is not int or meses <= 0:
        raise ValueError("shift_months deve ser um inteiro positivo para avancar o periodo.")
    if int(config["drift"]["seed"]) < 0:
        raise ValueError("A seed deve ser nao negativa.")
    jitter = float(config["drift"].get("jitter", 0.0))
    if not math.isfinite(jitter) or jitter < 0:
        raise ValueError("O jitter deve ser finito e nao negativo.")
    config["drift"]["jitter"] = jitter

    variaveis = config["drift"].get("variables", {})
    if not isinstance(variaveis, dict):
        raise ValueError("As variaveis de drift devem formar um objeto JSON.")
    if len(variaveis) < MINIMO_VARIAVEIS:
        raise ValueError(
            f"A simulacao de Data Drift exige ao menos {MINIMO_VARIAVEIS} variaveis alteradas."
        )
    for coluna, especificacao in variaveis.items():
        if not isinstance(especificacao, dict):
            raise ValueError(f"Especificacao invalida para {coluna}.")
        operacao = especificacao.get("operation")
        if operacao not in OPERACOES:
            raise ValueError(
                f"Operacao invalida para {coluna}: {operacao}. Esperado: {list(OPERACOES)}."
            )
        if operacao == "scale" and "factor" not in especificacao:
            raise ValueError(f"Operacao de escala exige 'factor' em {coluna}.")
        if operacao == "shift" and "delta" not in especificacao:
            raise ValueError(f"Operacao de deslocamento exige 'delta' em {coluna}.")
        if len(especificacao.get("clip", [])) != 2:
            raise ValueError(f"Limite de faixa invalido para {coluna}.")
    return config
