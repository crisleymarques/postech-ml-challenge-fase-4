"""Preservacao do contrato ao extrair as operacoes de manifesto."""

from pathlib import Path

import pytest

from monitoramento.infraestrutura import manifestos


class ErroConsumidor(Exception):
    """Excecao especifica de uma etapa consumidora."""


def test_manifesto_propaga_excecao_do_consumidor(tmp_path: Path) -> None:
    with pytest.raises(ErroConsumidor, match="Manifesto de entrada nao encontrado"):
        manifestos.carregar_manifesto(tmp_path / "ausente.json", erro=ErroConsumidor)


def test_artefato_preserva_contrato_de_hash_opcional(tmp_path: Path) -> None:
    caminho = tmp_path / "artefato.csv"
    caminho.write_text("a\n1\n")

    assert manifestos.verificar_artefato({"path": str(caminho)}, erro=ErroConsumidor) == caminho
    with pytest.raises(ErroConsumidor, match="Hash divergente"):
        manifestos.verificar_artefato(
            {"path": str(caminho), "sha256": "0" * 64}, erro=ErroConsumidor
        )
