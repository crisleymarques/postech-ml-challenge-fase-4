"""Orquestracao da coleta e publicacao da saude operacional."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monitoramento.infraestrutura.observabilidade import agora_iso

from .catalogo import (
    CONTRACT_VERSION,
    ESCOPO_AGREGADO,
    ESCOPO_EXECUCAO,
    carregar_catalogo,
    metricas_por_escopo,
)
from .entradas import EVENTO_ALERTA, STATUS_SUCESSO, carregar_contexto, listar_execucoes
from .medicao import medir_agregado, medir_execucao
from .resultados import gravar_saude, montar_resumo


def gerar(runs_dir: Path, output_dir: Path, catalogo_path: Path) -> dict[str, Any]:
    """Coleta as metricas de saude e publica `saude.json`."""
    catalogo = carregar_catalogo(catalogo_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    execucoes_saida: list[dict[str, Any]] = []
    eventos: list[dict[str, Any]] = []
    resumos: list[dict[str, Any]] = []
    for run_dir in listar_execucoes(runs_dir):
        contexto, eventos_run = carregar_contexto(run_dir)
        manifesto = contexto["manifesto"]
        publicado = manifesto["status"] == STATUS_SUCESSO
        stage = contexto["stage"]
        medidas = [
            medida
            for medida in (
                medir_execucao(metrica, contexto)
                for metrica in metricas_por_escopo(catalogo, ESCOPO_EXECUCAO)
            )
            if medida is not None
        ]
        run_id = manifesto.get("run_id") or run_dir.name
        execucoes_saida.append(
            {
                "run_id": run_id,
                "stage": stage,
                "status": manifesto.get("status"),
                "metricas": medidas,
            }
        )
        resumos.append({"run_id": run_id, "stage": stage, "status": manifesto.get("status")})
        eventos.extend(e for e in eventos_run if publicado or e.get("event") != EVENTO_ALERTA)

    contexto_agregado = {
        "execucoes": resumos,
        "eventos": eventos,
        "output_dir": output_dir,
    }
    agregados = [
        medir_agregado(metrica, contexto_agregado)
        for metrica in metricas_por_escopo(catalogo, ESCOPO_AGREGADO)
    ]

    saude = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": agora_iso(),
        "runs_dir": runs_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "catalog_path": catalogo_path.as_posix(),
        "execucoes": execucoes_saida,
        "agregados": agregados,
        "resumo": montar_resumo(execucoes_saida, agregados),
    }
    gravar_saude(output_dir / "saude.json", saude)
    return saude
