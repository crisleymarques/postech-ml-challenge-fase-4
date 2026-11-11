"""Catalogo e aplicabilidade das metricas de saude."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "1.0"

ESCOPO_EXECUCAO = "execucao"

ESCOPO_AGREGADO = "agregado"

ESCOPOS = (ESCOPO_EXECUCAO, ESCOPO_AGREGADO)

CAMPOS_METRICA = ("id", "categoria", "nome", "unidade", "escopo", "coleta", "criterio")


def versao_maior(versao: Any) -> str:
    """Extrai o componente principal de uma versao de contrato."""
    return str(versao or "").split(".")[0]


def carregar_catalogo(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente o catalogo de metricas de saude."""
    if not caminho.is_file():
        raise ValueError(f"Catalogo nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        catalogo = json.load(arquivo)

    if versao_maior(catalogo.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato do catalogo incompativel.")

    metricas = catalogo.get("metricas")
    if not isinstance(metricas, list) or not metricas:
        raise ValueError("O catalogo deve declarar ao menos uma metrica.")

    ids: list[str] = []
    for metrica in metricas:
        if not isinstance(metrica, dict):
            raise ValueError(f"Metrica invalida no catalogo: {metrica}")
        ausentes = [campo for campo in CAMPOS_METRICA if not metrica.get(campo)]
        if ausentes:
            raise ValueError(f"Metrica {metrica.get('id')} sem campos: {ausentes}")
        if metrica["escopo"] not in ESCOPOS:
            raise ValueError(f"Escopo invalido na metrica {metrica['id']}: {metrica['escopo']}")
        etapas = metrica.get("etapas")
        if etapas is not None and (not isinstance(etapas, list) or not etapas):
            raise ValueError(f"'etapas' invalido na metrica {metrica['id']}.")
        ids.append(metrica["id"])

    if len(set(ids)) != len(ids):
        raise ValueError("Ids de metrica duplicados no catalogo.")
    return catalogo


def metricas_por_escopo(catalogo: dict[str, Any], escopo: str) -> list[dict[str, Any]]:
    """Metricas do catalogo com o escopo informado, na ordem declarada."""
    return [metrica for metrica in catalogo["metricas"] if metrica["escopo"] == escopo]


def aplicavel(metrica: dict[str, Any], stage: str | None) -> bool:
    """Indica se a metrica se aplica a etapa informada."""
    etapas = metrica.get("etapas")
    return etapas is None or stage in etapas
