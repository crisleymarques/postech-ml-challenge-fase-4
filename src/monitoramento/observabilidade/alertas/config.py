"""Config da integracao de alertas operacionais."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .entradas import SITUACAO_ATENCAO, SITUACAO_CRITICO, ler_json, validar_versao

SEVERIDADE_WARNING = "warning"


SEVERIDADE_CRITICAL = "critical"


ORDEM_SEVERIDADES = (SEVERIDADE_WARNING, SEVERIDADE_CRITICAL)


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida os criterios de alerta suportados pelos consumidores."""
    if not caminho.is_file():
        raise ValueError(f"Configuracao de alertas nao encontrada: {caminho}")
    config = ler_json(caminho, "Configuracao de alertas")
    validar_versao(config, "Configuracao de alertas")

    severidades = config.get("severidades")
    if not isinstance(severidades, list) or not severidades:
        raise ValueError("A configuracao deve declarar as severidades.")
    if SEVERIDADE_CRITICAL not in severidades:
        raise ValueError(f"A severidade '{SEVERIDADE_CRITICAL}' deve constar na configuracao.")
    if sorted(severidades, key=str) != sorted(ORDEM_SEVERIDADES):
        raise ValueError("As severidades suportadas sao warning e critical, sem duplicatas.")

    mapa = config.get("severidade_por_situacao")
    if (
        not isinstance(mapa, dict)
        or set(mapa) != {SITUACAO_ATENCAO, SITUACAO_CRITICO}
        or any(valor not in severidades for valor in mapa.values())
    ):
        raise ValueError("Mapeamento 'severidade_por_situacao' invalido ou incompleto.")

    for chave in ("severidade_falha", "severidade_alerta_drift"):
        if config.get(chave) not in severidades:
            raise ValueError(f"Campo '{chave}' deve referenciar uma severidade declarada.")
    return config
