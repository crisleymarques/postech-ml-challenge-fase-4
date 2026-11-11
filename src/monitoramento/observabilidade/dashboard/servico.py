"""Carregamento coordenado dos dados do painel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .entradas import carregar_execucoes, ler_alertas, ler_json
from .indicadores import resumir_execucoes


def carregar_painel(
    runs_dir: Path,
    logs_path: Path,
    saude_path: Path,
    alertas_path: Path,
) -> dict[str, Any]:
    """Carrega as execucoes e os artefatos consolidados para o painel."""
    execucoes = carregar_execucoes(runs_dir)
    logs = ler_json(logs_path)
    saude = ler_json(saude_path)
    alertas = ler_alertas(alertas_path)
    return {
        "execucoes": execucoes,
        "resumo_execucoes": resumir_execucoes(execucoes),
        "logs": logs,
        "saude": saude,
        "alertas": alertas,
        "fontes": {
            "runs_dir": runs_dir.as_posix(),
            "logs": logs_path.as_posix(),
            "saude": saude_path.as_posix(),
            "alertas": alertas_path.as_posix(),
        },
        "disponibilidade": {
            "logs": logs is not None,
            "saude": saude is not None,
            "alertas": alertas is not None,
        },
    }
