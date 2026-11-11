"""Testes e apoio de test entradas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.observabilidade.logs import classificacao, entradas

from .apoio import escrever_execucao, evento


def test_listar_execucoes_ignora_diretorios_sem_eventos(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    escrever_execucao(
        runs_dir, "run-b", [evento(run_id="run-b", stage="treinamento", tipo="started")]
    )
    escrever_execucao(
        runs_dir, "run-a", [evento(run_id="run-a", stage="treinamento", tipo="started")]
    )
    (runs_dir / "run-sem-eventos").mkdir()

    execucoes = entradas.listar_execucoes(runs_dir)

    assert [caminho.name for caminho in execucoes] == ["run-a", "run-b"]


def test_listar_execucoes_diretorio_ausente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        entradas.listar_execucoes(tmp_path / "inexistente")


def test_ler_eventos_ignora_linhas_vazias(tmp_path: Path) -> None:
    caminho = tmp_path / "events.jsonl"
    caminho.write_text(
        json.dumps(evento(run_id="run-a", stage="preparacao", tipo="started")) + "\n\n",
        encoding="utf-8",
    )

    eventos = entradas.ler_eventos(caminho)

    assert len(eventos) == 1
    assert eventos[0]["stage"] == "preparacao"


def test_ler_eventos_rejeita_json_invalido(tmp_path: Path) -> None:
    caminho = tmp_path / "events.jsonl"
    caminho.write_text("{nao-eh-json}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        entradas.ler_eventos(caminho)


def test_ler_manifesto_ausente_retorna_none(tmp_path: Path) -> None:
    assert entradas.ler_manifesto(tmp_path / "manifest.json") is None


def test_ler_manifesto_invalido_retorna_none(tmp_path: Path) -> None:
    caminho = tmp_path / "manifest.json"
    caminho.write_text("{invalido}", encoding="utf-8")

    assert entradas.ler_manifesto(caminho) is None


def test_analisar_execucao_usa_manifesto(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    eventos = [evento(run_id="run-a", stage="preparacao", tipo="completed")]
    manifesto = {
        "contract_version": "1.0",
        "run_id": "run-a",
        "stage": "preparacao",
        "status": "success",
        "error_code": None,
        "transformations": ["trim_espacos", "binarizar_alvo"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifesto), encoding="utf-8")

    info = entradas.analisar_execucao(run_dir, eventos)

    assert info["status"] == "success"
    assert info["stage"] == "preparacao"
    assert info["resultado"] == "sucesso"
    assert info["transformations"] == ["trim_espacos", "binarizar_alvo"]


def test_analisar_execucao_completed_sem_manifesto_e_incompleta(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    eventos = [evento(run_id="run-a", stage="preparacao", tipo="completed")]

    info = entradas.analisar_execucao(run_dir, eventos)

    assert info["status"] == "incomplete"
    assert info["stage"] == "preparacao"
    assert info["transformations"] == []


def test_analisar_execucao_sem_manifesto_infere_incompleto(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    eventos = [evento(run_id="run-a", stage="treinamento", tipo="started")]

    info = entradas.analisar_execucao(run_dir, eventos)

    assert info["status"] == classificacao.STATUS_INCOMPLETO
    assert info["run_id"] == "run-a"
    assert info["categoria"] == "treinamento"


@pytest.mark.parametrize("eventos", [[], ["started"], ["completed"]])
def test_manifesto_antecipado_ou_evento_divergente_nao_confirma_sucesso(tmp_path, eventos):
    run_dir = escrever_execucao(
        tmp_path,
        "run-a",
        [evento(run_id="outro-run", stage="deteccao", tipo=tipo) for tipo in eventos],
        {"contract_version": "1.0", "run_id": "run-a", "stage": "deteccao", "status": "success"},
    )
    info = entradas.analisar_execucao(run_dir, entradas.ler_eventos(run_dir / "events.jsonl"))
    assert info["status"] == "incomplete"
