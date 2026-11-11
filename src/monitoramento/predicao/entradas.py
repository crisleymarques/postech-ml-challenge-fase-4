"""Resolucao e leitura das entradas especificas da predicao."""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monitoramento.infraestrutura import manifestos as manifestos_infra
from monitoramento.infraestrutura import qualidade

from .config import CONTRACT_VERSION, ErroPredicao

carregar_manifesto = partial(
    manifestos_infra.carregar_manifesto, erro=ErroPredicao, contract_version=CONTRACT_VERSION
)
artefato = partial(manifestos_infra.artefato, erro=ErroPredicao)


def verificar_artefato(item: dict[str, Any]) -> Path:
    """Exige um SHA-256 valido antes de consumir qualquer artefato."""
    esperado = item.get("sha256")
    if not isinstance(esperado, str) or re.fullmatch(r"[0-9a-fA-F]{64}", esperado) is None:
        raise ErroPredicao("Descritor sem sha256 valido.")
    return manifestos_infra.verificar_artefato(
        {**item, "sha256": esperado.lower()}, erro=ErroPredicao
    )


def separar_manifestos(
    manifestos: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Identifica o manifesto do modelo e o do lote pela etapa produtora."""
    etapa_modelo = config["model"]["producer_stage"]
    etapa_lote = config["batch"]["producer_stage"]
    etapa_validacao = config["validation"]["producer_stage"]
    esperadas = (etapa_modelo, etapa_lote, etapa_validacao)

    encontrados: dict[str, dict[str, Any]] = {}
    for manifesto in manifestos:
        if manifesto.get("status") != "success":
            raise ErroPredicao("Dependencia nao concluiu com sucesso.")
        if manifestos_infra.versao_maior(manifesto.get("contract_version")) != "1":
            raise ErroPredicao("Versao principal do contrato do manifesto incompativel.")
        etapa = manifesto.get("stage")
        if etapa not in esperadas:
            raise ErroPredicao(f"Etapa de entrada inesperada: {etapa}.")
        if etapa in esperadas:
            if etapa in encontrados:
                raise ErroPredicao(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroPredicao(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_modelo], encontrados[etapa_lote], encontrados[etapa_validacao]


def verificar_compatibilidade(
    config: dict[str, Any], manifesto_modelo: dict[str, Any], manifesto_lote: dict[str, Any]
) -> None:
    """Confere que modelo e lote pertencem ao mesmo dataset da configuracao."""
    dataset_id = config["dataset_id"]
    for manifesto in (manifesto_modelo, manifesto_lote):
        if manifesto.get("dataset_id") != dataset_id:
            raise ErroPredicao(
                f"dataset_id divergente no manifesto {manifesto.get('run_id')}: "
                f"{manifesto.get('dataset_id')} != {dataset_id}"
            )


def ler_lote(caminho: Path) -> pd.DataFrame:
    """Le o lote como texto tratando campo vazio como ausencia, como no treino."""
    base = pd.read_csv(caminho, dtype=str, na_filter=False)
    return base.replace("", np.nan)


verificar_metadata = partial(qualidade.verificar_metadata, erro=ErroPredicao)
verificar_aprovacao = partial(qualidade.verificar_aprovacao, erro=ErroPredicao)
