"""Resolucao e leitura das entradas especificas da predicao."""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monitoramento.infraestrutura import manifestos as manifestos_infra

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


def verificar_metadata(
    metadata: dict[str, Any], modelo: dict[str, Any], lote: dict[str, Any]
) -> None:
    """Exige identidade e schema consistentes antes de carregar o modelo."""
    schema = modelo.get("schema_version")
    if (
        not isinstance(schema, str)
        or not schema.strip()
        or any(origem.get("schema_version") != schema for origem in (metadata, lote))
    ):
        raise ErroPredicao("schema_version ausente ou divergente entre modelo, metadados e lote.")
    for campo in ("dataset_id", "model_id", "reference_id"):
        valor = modelo.get(campo)
        if not isinstance(valor, str) or not valor.strip() or metadata.get(campo) != valor:
            raise ErroPredicao(f"{campo} ausente ou divergente nos metadados do modelo.")


def ler_lote(caminho: Path) -> pd.DataFrame:
    """Le o lote como texto tratando campo vazio como ausencia, como no treino."""
    base = pd.read_csv(caminho, dtype=str, na_filter=False)
    return base.replace("", np.nan)


def verificar_aprovacao(
    validacao: dict[str, Any], lote: dict[str, Any], features: dict[str, Any]
) -> None:
    """Vincula a aprovacao de producao ao lote e ao CSV exato consumido."""
    qualidade = validacao.get("quality")
    if not validacao.get("run_id") or not isinstance(qualidade, dict):
        raise ErroPredicao("Manifesto de validacao sem aprovacao de qualidade.")
    if qualidade.get("approved") is not True:
        raise ErroPredicao("Qualidade do lote nao aprovada.")
    for campo in ("dataset_id", "schema_version", "scenario_id", "batch_id"):
        valor = lote.get(campo)
        if not isinstance(valor, str) or not valor.strip() or validacao.get(campo) != valor:
            raise ErroPredicao(f"{campo} divergente na aprovacao do lote.")
        if campo in ("dataset_id", "schema_version") and qualidade.get(campo) != valor:
            raise ErroPredicao(f"{campo} divergente no resumo de qualidade.")
    entradas = [i for i in validacao.get("inputs", []) if i.get("artifact_type") == "features"]
    aprovados = [i for i in qualidade.get("artifacts", []) if i.get("artifact_type") == "features"]
    if len(entradas) != 1 or len(aprovados) != 1:
        raise ErroPredicao("Aprovacao deve identificar um unico artefato features.")
    entrada, aprovado = entradas[0], aprovados[0]
    if (
        aprovado.get("approved") is not True
        or aprovado.get("sha256") != features.get("sha256")
        or entrada.get("sha256") != features.get("sha256")
        or not isinstance(entrada.get("path"), str)
        or Path(entrada["path"]).resolve() != Path(features["path"]).resolve()
        or not lote.get("run_id")
        or entrada.get("producer_run_id") != lote["run_id"]
        or features.get("producer_run_id") != lote["run_id"]
    ):
        raise ErroPredicao("Aprovacao nao corresponde ao hash, caminho ou produtor das features.")
