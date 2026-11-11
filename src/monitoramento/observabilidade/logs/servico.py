"""Coordenacao da consolidacao dos logs operacionais."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .classificacao import classificar_evento
from .entradas import analisar_execucao, ler_eventos, listar_execucoes
from .resultados import gravar_operacional, gravar_resumo
from .resumo import montar_resumo


def gerar(runs_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Consolida os logs de todas as execucoes e publica os artefatos."""
    execucoes: list[dict[str, Any]] = []
    eventos: list[dict[str, Any]] = []
    for run_dir in listar_execucoes(runs_dir):
        eventos_run = ler_eventos(run_dir / "events.jsonl")
        execucoes.append(analisar_execucao(run_dir, eventos_run))
        for evento in eventos_run:
            classificado = classificar_evento(evento)
            classificado["run_dir"] = run_dir.as_posix()
            eventos.append(classificado)

    eventos.sort(key=lambda item: (item.get("timestamp") or "", item.get("run_id") or ""))
    resumo = montar_resumo(
        runs_dir=runs_dir,
        output_dir=output_dir,
        execucoes=execucoes,
        eventos=eventos,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    gravar_operacional(output_dir / "operacional.jsonl", eventos)
    gravar_resumo(output_dir / "resumo.json", resumo)
    return resumo
