"""Leitura e integridade de manifestos, sem regras especificas das etapas."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .arquivos import sha256_arquivo


def versao_maior(versao: Any) -> str:
    """Extrai o componente principal de uma versao de contrato."""
    return str(versao or "").split(".")[0]


def carregar_manifesto(
    caminho: Path, *, erro: type[Exception], contract_version: str = "1.0"
) -> dict[str, Any]:
    """Le um manifesto de entrada e exige o produtor com sucesso."""
    if not caminho.is_file():
        raise erro(f"Manifesto de entrada nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        manifesto = json.load(arquivo)

    if manifesto.get("status") != "success":
        raise erro(f"Produtor {manifesto.get('run_id')} nao concluiu com sucesso.")
    if versao_maior(manifesto.get("contract_version")) != versao_maior(contract_version):
        raise erro("Versao principal do contrato do manifesto incompativel.")
    return manifesto


def artefato(manifesto: dict[str, Any], tipo: str, *, erro: type[Exception]) -> dict[str, Any]:
    """Devolve o descritor de um artefato de saida pelo tipo."""
    for item in manifesto.get("outputs", []):
        if item.get("artifact_type") == tipo:
            return item
    raise erro(f"Artefato '{tipo}' ausente no manifesto {manifesto.get('run_id')}.")


def verificar_artefato(
    item: dict[str, Any], *, erro: type[Exception], hash_obrigatorio: bool = False
) -> Path:
    """Confere existencia e integridade de um artefato antes de consumi-lo."""
    caminho = Path(item.get("path", ""))
    if not caminho.is_file():
        raise erro(f"Artefato nao encontrado: {caminho}")
    esperado = item.get("sha256")
    if hash_obrigatorio:
        if not isinstance(esperado, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", esperado):
            raise erro(f"SHA-256 ausente ou invalido para o artefato: {caminho}")
        esperado = esperado.lower()
    if esperado and sha256_arquivo(caminho) != esperado:
        raise erro(f"Hash divergente para o artefato: {caminho}")
    return caminho
