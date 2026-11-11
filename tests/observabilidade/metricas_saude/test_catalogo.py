"""Testes de catalogo das metricas de saude."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.observabilidade.metricas_saude import catalogo as cat

from .apoio import CATALOGO_REAL, metrica


def test_carregar_catalogo_real() -> None:
    catalogo = cat.carregar_catalogo(CATALOGO_REAL)

    assert catalogo["contract_version"] == "1.0"
    assert len(catalogo["metricas"]) == 15
    assert set(catalogo["categorias"]) == {
        "pipeline",
        "qualidade",
        "drift",
        "modelo",
        "infraestrutura",
    }
    ids = {item["id"] for item in catalogo["metricas"]}
    assert "pipeline.duracao_s" in ids
    assert "drift.alertas_total" in ids


def test_carregar_catalogo_ausente(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cat.carregar_catalogo(tmp_path / "inexistente.json")


def test_carregar_catalogo_sem_metricas(tmp_path: Path) -> None:
    caminho = tmp_path / "cat.json"
    caminho.write_text(json.dumps({"contract_version": "1.0", "metricas": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        cat.carregar_catalogo(caminho)


def test_carregar_catalogo_escopo_invalido(tmp_path: Path) -> None:
    caminho = tmp_path / "cat.json"
    catalogo = {
        "contract_version": "1.0",
        "metricas": [metrica(escopo="global")],
    }
    caminho.write_text(json.dumps(catalogo), encoding="utf-8")

    with pytest.raises(ValueError):
        cat.carregar_catalogo(caminho)


def test_carregar_catalogo_id_duplicado(tmp_path: Path) -> None:
    caminho = tmp_path / "cat.json"
    catalogo = {
        "contract_version": "1.0",
        "metricas": [metrica(id="a.b"), metrica(id="a.b")],
    }
    caminho.write_text(json.dumps(catalogo), encoding="utf-8")

    with pytest.raises(ValueError):
        cat.carregar_catalogo(caminho)


def test_carregar_catalogo_versao_incompativel(tmp_path: Path) -> None:
    caminho = tmp_path / "cat.json"
    catalogo = {"contract_version": "2.0", "metricas": [metrica()]}
    caminho.write_text(json.dumps(catalogo), encoding="utf-8")

    with pytest.raises(ValueError):
        cat.carregar_catalogo(caminho)


def test_metricas_por_escopo() -> None:
    catalogo = {
        "metricas": [
            metrica(id="a", escopo="execucao"),
            metrica(id="b", escopo="agregado"),
            metrica(id="c", escopo="execucao"),
        ]
    }

    execucao = cat.metricas_por_escopo(catalogo, "execucao")
    agregado = cat.metricas_por_escopo(catalogo, "agregado")

    assert [item["id"] for item in execucao] == ["a", "c"]
    assert [item["id"] for item in agregado] == ["b"]


def test_aplicavel() -> None:
    todas = metrica(etapas=None)
    restrita = metrica(etapas=["deteccao", "analise"])

    assert cat.aplicavel(todas, "treinamento") is True
    assert cat.aplicavel(todas, None) is True
    assert cat.aplicavel(restrita, "deteccao") is True
    assert cat.aplicavel(restrita, "treinamento") is False
