"""Testes de entradas do painel."""

from __future__ import annotations

from pathlib import Path

import pytest

from monitoramento.observabilidade.dashboard import config, entradas
from tests.observabilidade.dashboard.apoio import escrever_execucao, evento


def test_ler_json_ausente(tmp_path: Path) -> None:
    assert entradas.ler_json(tmp_path / "inexistente.json") is None


def test_ler_json_invalido(tmp_path: Path) -> None:
    caminho = tmp_path / "saude.json"
    caminho.write_text("{nao e json", encoding="utf-8")
    assert entradas.ler_json(caminho) is None


def test_ler_json_lista_retorna_none(tmp_path: Path) -> None:
    caminho = tmp_path / "lista.json"
    caminho.write_text("[1, 2, 3]", encoding="utf-8")
    assert entradas.ler_json(caminho) is None


def test_ler_eventos_tolerante(tmp_path: Path) -> None:
    caminho = tmp_path / "events.jsonl"
    caminho.write_text('{"event": "started"}\n\nnao e json\n{"event": "completed"}\n', "utf-8")
    eventos = entradas.ler_eventos(caminho)
    assert [evento["event"] for evento in eventos] == ["started", "completed"]


def test_inferir_status_usa_ultimo_terminal() -> None:
    eventos = [
        evento("run-a", "validacao", "started"),
        evento("run-a", "validacao", "completed"),
        evento("run-a", "validacao", "drift_alert"),
    ]
    assert entradas.inferir_status(eventos) == config.STATUS_SUCESSO


def test_inferir_status_sem_terminal_e_incompleto() -> None:
    assert (
        entradas.inferir_status([evento("run-a", "validacao", "started")])
        == config.STATUS_INCOMPLETO
    )


def test_listar_execucoes_ignora_diretorio_sem_artefato(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    escrever_execucao(runs_dir, "run-b", manifesto={"run_id": "run-b"})
    escrever_execucao(runs_dir, "run-a", eventos=[evento("run-a", "preparacao", "started")])
    (runs_dir / "vazio").mkdir()
    nomes = [caminho.name for caminho in entradas.listar_execucoes(runs_dir)]
    assert nomes == ["run-a", "run-b"]


def test_listar_execucoes_diretorio_inexistente(tmp_path: Path) -> None:
    assert entradas.listar_execucoes(tmp_path / "sem-runs") == []


def test_carregar_execucoes_com_manifesto(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    escrever_execucao(
        runs_dir,
        "run-a",
        manifesto={
            "contract_version": "1.0",
            "run_id": "run-a",
            "stage": "treinamento",
            "status": "success",
            "started_at": "2026-11-09T00:00:00.000000Z",
            "finished_at": "2026-11-09T00:00:05.000000Z",
        },
        eventos=[evento("run-a", "treinamento", "completed")],
    )
    execucoes = entradas.carregar_execucoes(runs_dir)

    assert len(execucoes) == 1
    assert execucoes[0]["run_id"] == "run-a"
    assert execucoes[0]["stage"] == "treinamento"
    assert execucoes[0]["status"] == "success"
    assert execucoes[0]["event_count"] == 1
    assert execucoes[0]["incompleta"] is False


def test_carregar_execucoes_incompleta(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    escrever_execucao(
        runs_dir,
        "run-b",
        eventos=[evento("run-b", "predicao", "started")],
    )
    execucoes = entradas.carregar_execucoes(runs_dir)

    assert execucoes[0]["status"] == config.STATUS_INCOMPLETO
    assert execucoes[0]["incompleta"] is True
    assert execucoes[0]["stage"] == "predicao"


def test_eventos_invalidos_entre_validos_preservam_eventos_validos(tmp_path):
    caminho = tmp_path / "events.jsonl"
    caminho.write_text(
        'null\n{"event":"started"}\n[]\nnao-json\n{"event":"completed"}\n',
        encoding="utf-8",
    )
    assert [e["event"] for e in entradas.ler_eventos(caminho)] == ["started", "completed"]


@pytest.mark.parametrize(
    "erro", [PermissionError("sem acesso"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalido")]
)
def test_falha_de_leitura_de_eventos_isolada(tmp_path, monkeypatch, erro):
    runs = tmp_path / "runs"
    ruim = escrever_execucao(runs, "ruim", eventos=[evento("ruim", "predicao", "started")])
    escrever_execucao(runs, "outra", manifesto={"status": "failed", "stage": "predicao"})
    original = Path.open

    def abrir(caminho, *args, **kwargs):
        if caminho == ruim / "events.jsonl":
            raise erro
        return original(caminho, *args, **kwargs)

    monkeypatch.setattr(Path, "open", abrir)
    execucoes = entradas.carregar_execucoes(runs)
    assert [e["status"] for e in execucoes] == ["failed", "incomplete"]
