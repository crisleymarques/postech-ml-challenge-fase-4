"""Resolucao e compatibilidade das entradas da deteccao."""

from __future__ import annotations

from functools import partial
from typing import Any

from monitoramento.infraestrutura import manifestos

from .config import CONTRACT_VERSION, ErroDeteccao

carregar_manifesto = partial(
    manifestos.carregar_manifesto, erro=ErroDeteccao, contract_version=CONTRACT_VERSION
)
artefato = partial(manifestos.artefato, erro=ErroDeteccao)
verificar_artefato = partial(manifestos.verificar_artefato, erro=ErroDeteccao)


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
                raise ErroDeteccao(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroDeteccao(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_ref], encontrados[etapa_lote]


def verificar_compatibilidade(
    config: dict[str, Any], manifesto_ref: dict[str, Any], manifesto_lote: dict[str, Any]
) -> None:
    """Confere que referencia e lote pertencem ao mesmo dataset da configuracao."""
    dataset_id = config["dataset_id"]
    for manifesto in (manifesto_ref, manifesto_lote):
        if manifesto.get("dataset_id") != dataset_id:
            raise ErroDeteccao(
                f"dataset_id divergente no manifesto {manifesto.get('run_id')}: "
                f"{manifesto.get('dataset_id')} != {dataset_id}"
            )
