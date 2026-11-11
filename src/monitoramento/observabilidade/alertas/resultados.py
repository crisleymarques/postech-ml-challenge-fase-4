"""Resultados da integracao de alertas operacionais."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ORDEM_SEVERIDADES, SEVERIDADE_CRITICAL
from .regras import ORIGEM_LOGS, ORIGEM_SAUDE


def montar_resumo(alertas: list[dict[str, Any]]) -> dict[str, Any]:
    """Consolida as contagens de alertas por severidade, categoria e origem."""
    por_severidade = {severidade: 0 for severidade in ORDEM_SEVERIDADES}
    por_categoria: dict[str, int] = {}
    por_origem = {ORIGEM_SAUDE: 0, ORIGEM_LOGS: 0}
    for alerta in alertas:
        severidade = alerta["severidade"]
        por_severidade[severidade] = por_severidade.get(severidade, 0) + 1
        categoria = alerta.get("categoria") or "desconhecida"
        por_categoria[categoria] = por_categoria.get(categoria, 0) + 1
        origem = alerta.get("fonte")
        por_origem[origem] = por_origem.get(origem, 0) + 1
    return {
        "total": len(alertas),
        "por_severidade": por_severidade,
        "por_categoria": dict(sorted(por_categoria.items())),
        "por_origem": por_origem,
        "criticos": [
            alerta["id"] for alerta in alertas if alerta["severidade"] == SEVERIDADE_CRITICAL
        ],
    }


def gravar_alertas(caminho: Path, relatorio: dict[str, Any]) -> None:
    """Grava o relatorio integrado de alertas em JSON UTF-8."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
