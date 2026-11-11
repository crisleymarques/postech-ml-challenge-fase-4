"""Testes migrados da etapa de predicao."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.predicao import config as cfg
from monitoramento.predicao import entradas

from .apoio import (
    montar_config,
)


def test_carregar_manifesto_rejeita_status_nao_sucesso(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.json"
    caminho.write_text(
        json.dumps({"contract_version": "1.0", "status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(cfg.ErroPredicao, match="nao concluiu com sucesso"):
        entradas.carregar_manifesto(caminho)


def test_separar_manifestos_rejeita_ausente() -> None:
    manifesto = {
        "stage": "treinamento",
        "run_id": "r1",
        "status": "success",
        "contract_version": "1.0",
    }

    with pytest.raises(cfg.ErroPredicao, match="Manifestos ausentes"):
        entradas.separar_manifestos([manifesto], montar_config())


def test_separar_manifestos_rejeita_duplicado() -> None:
    duplicados = [
        {"stage": "treinamento", "run_id": "r1", "status": "success", "contract_version": "1.0"},
        {"stage": "treinamento", "run_id": "r2", "status": "success", "contract_version": "1.0"},
        {"stage": "simulacao", "run_id": "r3", "status": "success", "contract_version": "1.0"},
    ]

    with pytest.raises(cfg.ErroPredicao, match="duplicado"):
        entradas.separar_manifestos(duplicados, montar_config())


def test_verificar_artefato_rejeita_hash_divergente(tmp_path: Path) -> None:
    caminho = tmp_path / "artefato.csv"
    caminho.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(cfg.ErroPredicao, match="Hash divergente"):
        entradas.verificar_artefato({"path": caminho.as_posix(), "sha256": "0" * 64})
