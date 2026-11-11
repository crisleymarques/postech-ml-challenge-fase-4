"""Resolucao e alinhamento das entradas da avaliacao."""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monitoramento.infraestrutura import manifestos, qualidade
from monitoramento.infraestrutura.manifestos import versao_maior
from monitoramento.infraestrutura.rotulos import validar_rotulos

from .config import CONTRACT_VERSION, IDENTIFICADOR, ErroAvaliacao

carregar_manifesto = partial(
    manifestos.carregar_manifesto, erro=ErroAvaliacao, contract_version=CONTRACT_VERSION
)
artefato = partial(manifestos.artefato, erro=ErroAvaliacao)
verificar_aprovacao = partial(qualidade.verificar_aprovacao, erro=ErroAvaliacao)
verificar_metadata = partial(qualidade.verificar_metadata, erro=ErroAvaliacao)


def verificar_artefato(item: dict[str, Any]) -> Path:
    """Exige hash valido e correspondente para cada entrada."""
    esperado = item.get("sha256")
    if not isinstance(esperado, str) or re.fullmatch(r"[0-9a-fA-F]{64}", esperado) is None:
        raise ErroAvaliacao("Descritor sem sha256 valido.")
    return manifestos.verificar_artefato({**item, "sha256": esperado.lower()}, erro=ErroAvaliacao)


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
            raise ErroAvaliacao("Dependencia nao concluiu com sucesso.")
        if versao_maior(manifesto.get("contract_version")) != "1":
            raise ErroAvaliacao("Versao principal do contrato incompativel.")
        etapa = manifesto.get("stage")
        if etapa not in esperadas:
            raise ErroAvaliacao("Etapa de entrada inesperada.")
        if etapa in esperadas:
            if etapa in encontrados:
                raise ErroAvaliacao(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroAvaliacao(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_modelo], encontrados[etapa_lote], encontrados[etapa_validacao]


def verificar_compatibilidade(
    config: dict[str, Any], manifesto_modelo: dict[str, Any], manifesto_lote: dict[str, Any]
) -> None:
    """Confere que modelo e lote pertencem ao mesmo dataset da configuracao."""
    dataset_id = config["dataset_id"]
    for manifesto in (manifesto_modelo, manifesto_lote):
        if manifesto.get("dataset_id") != dataset_id:
            raise ErroAvaliacao(
                f"dataset_id divergente no manifesto {manifesto.get('run_id')}: "
                f"{manifesto.get('dataset_id')} != {dataset_id}"
            )


def ler_lote(caminho: Path) -> pd.DataFrame:
    """Le o lote como texto tratando campo vazio como ausencia, como no treino."""
    base = pd.read_csv(caminho, dtype=str, na_filter=False)
    return base.replace("", np.nan)


def juntar_rotulos(identificadores: pd.Series, caminho: Path, alvo: str) -> np.ndarray:
    """Alinha os rotulos ao lote pelo identificador, preservando a ordem."""
    if not identificadores.is_unique:
        raise ErroAvaliacao("O lote possui identificadores repetidos.")
    rotulos = pd.read_csv(caminho, dtype=str, na_filter=False)
    if IDENTIFICADOR not in rotulos.columns:
        raise ErroAvaliacao(f"Identificador ausente nos rotulos: {IDENTIFICADOR}")
    if alvo not in rotulos.columns:
        raise ErroAvaliacao(f"Coluna alvo ausente nos rotulos: {alvo}")

    numericos = validar_rotulos(rotulos[alvo], erro=ErroAvaliacao)
    if not rotulos[IDENTIFICADOR].is_unique:
        raise ErroAvaliacao("Os rotulos possuem identificadores repetidos.")
    if identificadores.isna().any() or (identificadores == "").any():
        raise ErroAvaliacao("Identificadores ausentes no lote.")
    if set(rotulos[IDENTIFICADOR]) - set(identificadores):
        raise ErroAvaliacao("Os rotulos possuem registros extras ao lote.")

    mapa = pd.Series(numericos, index=rotulos[IDENTIFICADOR].to_numpy())
    alinhado = identificadores.map(mapa)
    if alinhado.isna().any():
        raise ErroAvaliacao("Rotulos ausentes para alguns registros do lote.")
    return alinhado.to_numpy().astype(int)
