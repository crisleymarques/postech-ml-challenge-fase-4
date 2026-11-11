"""Persistencia do log operacional e do resumo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def gravar_operacional(caminho: Path, eventos: list[dict[str, Any]]) -> None:
    """Grava o log operacional consolidado em JSON Lines, um evento por linha."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        for evento in eventos:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def gravar_resumo(caminho: Path, resumo: dict[str, Any]) -> None:
    """Grava o resumo consolidado em JSON UTF-8."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
