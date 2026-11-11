"""Testes de entradas dos alertas."""

from __future__ import annotations

from pathlib import Path

import pytest

from monitoramento.observabilidade.alertas import entradas as al


def test_ler_json_ausente(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nao encontrado"):
        al.ler_json(tmp_path / "saude.json", "Relatorio de saude")


def test_ler_json_invalido(tmp_path: Path) -> None:
    caminho = tmp_path / "saude.json"
    caminho.write_text("{nao e json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalido"):
        al.ler_json(caminho, "Relatorio de saude")
