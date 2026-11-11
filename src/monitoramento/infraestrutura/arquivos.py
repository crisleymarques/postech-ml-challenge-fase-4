"""Persistencia JSON e integridade de arquivos."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def sha256_arquivo(caminho: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em blocos."""
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def escrever_json(caminho: Path, conteudo: dict[str, Any]) -> None:
    """Grava um objeto JSON em UTF-8."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")


def escrever_json_atomico(caminho: Path, conteudo: dict[str, Any]) -> None:
    """Publica JSON completo sem expor um arquivo parcialmente escrito."""
    with NamedTemporaryFile(dir=caminho.parent, suffix=".tmp", delete=False) as arquivo:
        temporario = Path(arquivo.name)
    try:
        escrever_json(temporario, conteudo)
        os.replace(temporario, caminho)
    finally:
        temporario.unlink(missing_ok=True)
