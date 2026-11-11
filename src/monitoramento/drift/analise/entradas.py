"""Resolucao e compatibilidade das entradas da analise."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from monitoramento.infraestrutura import manifestos

from .config import CONTRACT_VERSION, ErroAnalise


def carregar_manifesto(entrada: Path | dict[str, Any]) -> dict[str, Any]:
    """Resolve caminhos e valida tambem manifestos recebidos pela API Python."""
    if isinstance(entrada, Path):
        return manifestos.carregar_manifesto(
            entrada, erro=ErroAnalise, contract_version=CONTRACT_VERSION
        )
    if not isinstance(entrada, dict) or entrada.get("status") != "success":
        raise ErroAnalise("Produtor nao concluiu com sucesso.")
    if manifestos.versao_maior(entrada.get("contract_version")) != manifestos.versao_maior(
        CONTRACT_VERSION
    ):
        raise ErroAnalise("Versao principal do contrato do manifesto incompativel.")
    return entrada


artefato = partial(manifestos.artefato, erro=ErroAnalise)
verificar_artefato = partial(manifestos.verificar_artefato, erro=ErroAnalise)


def separar_manifestos(
    manifestos: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identifica o manifesto da referencia e o do lote pela etapa produtora."""
    etapa_ref = config["reference"]["producer_stage"]
    etapa_lote = config["batch"]["producer_stage"]
    esperadas = (etapa_ref, etapa_lote)

    encontrados: dict[str, dict[str, Any]] = {}
    for manifesto in manifestos:
        etapa = manifesto.get("stage")
        if etapa in esperadas:
            if etapa in encontrados:
                raise ErroAnalise(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroAnalise(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_ref], encontrados[etapa_lote]


def verificar_compatibilidade(
    config: dict[str, Any], manifesto_ref: dict[str, Any], manifesto_lote: dict[str, Any]
) -> None:
    """Confere que referencia e lote pertencem ao mesmo dataset da configuracao."""
    dataset_id = config["dataset_id"]
    for manifesto in (manifesto_ref, manifesto_lote):
        if manifesto.get("dataset_id") != dataset_id:
            raise ErroAnalise(
                f"dataset_id divergente no manifesto {manifesto.get('run_id')}: "
                f"{manifesto.get('dataset_id')} != {dataset_id}"
            )


def validar_features_esperadas(
    referencia: pd.DataFrame, atual: pd.DataFrame, colunas: list[str]
) -> None:
    """Exige cobertura integral das features do modelo nas duas entradas."""
    for papel, base in (("referencia", referencia), ("lote", atual)):
        ausentes = [coluna for coluna in colunas if coluna not in base.columns]
        if ausentes:
            raise ErroAnalise(f"Features ausentes na entrada {papel}: {ausentes}")
