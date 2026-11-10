"""Resolucao e leitura das entradas especificas da predicao."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monitoramento.infraestrutura import manifestos

from .config import CONTRACT_VERSION, ErroPredicao

carregar_manifesto = partial(
    manifestos.carregar_manifesto, erro=ErroPredicao, contract_version=CONTRACT_VERSION
)
artefato = partial(manifestos.artefato, erro=ErroPredicao)
verificar_artefato = partial(manifestos.verificar_artefato, erro=ErroPredicao)


def separar_manifestos(
    manifestos: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identifica o manifesto do modelo e o do lote pela etapa produtora."""
    etapa_modelo = config["model"]["producer_stage"]
    etapa_lote = config["batch"]["producer_stage"]
    esperadas = (etapa_modelo, etapa_lote)

    encontrados: dict[str, dict[str, Any]] = {}
    for manifesto in manifestos:
        etapa = manifesto.get("stage")
        if etapa in esperadas:
            if etapa in encontrados:
                raise ErroPredicao(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroPredicao(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_modelo], encontrados[etapa_lote]


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
