"""Testes e apoio de test servico."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.observabilidade.logs import servico

from .apoio import escrever_execucao, evento


def test_manifesto_sem_arquivo_de_eventos_aparece_como_incompleto(tmp_path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"contract_version": "1.0", "run_id": "run-a", "stage": "deteccao", "status": "success"}
        )
    )
    resumo = servico.gerar(tmp_path, tmp_path / "logs")
    assert resumo["execucoes"]["por_status"] == {"incomplete": 1}


def test_gerar_ponta_a_ponta(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    escrever_execucao(
        runs_dir,
        "run-a",
        [
            evento(
                run_id="run-a",
                stage="preparacao",
                tipo="started",
                timestamp="2026-11-09T00:00:01.000000Z",
            ),
            evento(
                run_id="run-a",
                stage="preparacao",
                tipo="completed",
                timestamp="2026-11-09T00:00:02.000000Z",
            ),
        ],
        manifesto={
            "run_id": "run-a",
            "stage": "preparacao",
            "status": "success",
            "error_code": None,
            "transformations": ["trim_espacos", "binarizar_alvo"],
        },
    )
    escrever_execucao(
        runs_dir,
        "run-b",
        [
            evento(
                run_id="run-b",
                stage="validacao",
                tipo="started",
                timestamp="2026-11-09T00:00:03.000000Z",
            ),
            evento(
                run_id="run-b",
                stage="validacao",
                tipo="rejected",
                nivel="ERROR",
                error_code="QUALIDADE_REPROVADA",
                timestamp="2026-11-09T00:00:04.000000Z",
            ),
        ],
        manifesto={
            "run_id": "run-b",
            "stage": "validacao",
            "status": "rejected",
            "error_code": "QUALIDADE_REPROVADA",
        },
    )

    output_dir = tmp_path / "logs"
    resumo = servico.gerar(runs_dir, output_dir)

    assert (output_dir / "operacional.jsonl").is_file()
    assert (output_dir / "resumo.json").is_file()
    assert resumo["execucoes"]["total"] == 2
    assert resumo["eventos"]["total"] == 4
    assert len(resumo["falhas"]) == 1

    linhas = (output_dir / "operacional.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 4
    primeiro = json.loads(linhas[0])
    assert primeiro["categoria"] == "ingestao"
    assert primeiro["resultado"] == "inicio"


def test_gerar_ordena_eventos_por_timestamp(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    escrever_execucao(
        runs_dir,
        "run-b",
        [
            evento(
                run_id="run-b",
                stage="treinamento",
                tipo="started",
                timestamp="2026-11-09T00:00:05.000000Z",
            )
        ],
    )
    escrever_execucao(
        runs_dir,
        "run-a",
        [
            evento(
                run_id="run-a",
                stage="treinamento",
                tipo="started",
                timestamp="2026-11-09T00:00:01.000000Z",
            )
        ],
    )

    resumo = servico.gerar(runs_dir, tmp_path / "logs")

    linhas = (
        (tmp_path / "logs" / "operacional.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert json.loads(linhas[0])["run_id"] == "run-a"
    assert json.loads(linhas[1])["run_id"] == "run-b"
    assert resumo["execucoes"]["total"] == 2


def test_gerar_sem_execucoes(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    resumo = servico.gerar(runs_dir, tmp_path / "logs")

    assert resumo["execucoes"]["total"] == 0
    assert resumo["eventos"]["total"] == 0
    assert (tmp_path / "logs" / "operacional.jsonl").read_text(encoding="utf-8") == ""


def test_gerar_runs_dir_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        servico.gerar(tmp_path / "inexistente", tmp_path / "logs")


@pytest.mark.parametrize("status", ["failed", "rejected"])
@pytest.mark.parametrize("com_inicio", [False, True])
def test_gerar_preserva_falha_do_manifesto_sem_evento_terminal(
    tmp_path: Path, status: str, com_inicio: bool
) -> None:
    runs_dir = tmp_path / "runs"
    eventos = [evento(run_id="run-a", stage="validacao", tipo="started")] if com_inicio else []
    run_dir = escrever_execucao(
        runs_dir,
        "run-a",
        eventos,
        {
            "contract_version": "1.0",
            "run_id": "run-a",
            "stage": "validacao",
            "status": status,
            "error_code": "QUALIDADE_REPROVADA",
            "finished_at": "2026-11-09T00:00:01.000000Z",
        },
    )
    if not com_inicio:
        (run_dir / "events.jsonl").unlink()
    anteriores = {p.name: p.read_bytes() for p in run_dir.iterdir()}

    resumo = servico.gerar(runs_dir, tmp_path / "logs")

    assert resumo["execucoes"]["por_status"] == {status: 1}
    assert resumo["falhas"] == [
        {
            "run_id": "run-a",
            "stage": "validacao",
            "categoria": "validacao",
            "event": status,
            "level": "ERROR",
            "error_code": "QUALIDADE_REPROVADA",
            "timestamp": "2026-11-09T00:00:01.000000Z",
        }
    ]
    # O diagnostico do manifesto nao inventa eventos nem altera a fonte.
    assert resumo["eventos"]["total"] == len(eventos)
    linhas = (tmp_path / "logs" / "operacional.jsonl").read_text().splitlines()
    assert len(linhas) == len(eventos)
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == anteriores


@pytest.mark.parametrize("status", ["failed", "rejected"])
def test_gerar_nao_duplica_falha_presente_nos_eventos(tmp_path: Path, status: str) -> None:
    runs_dir = tmp_path / "runs"
    terminal = evento(
        run_id="run-a",
        stage="validacao",
        tipo=status,
        nivel="ERROR",
        error_code="QUALIDADE_REPROVADA",
    )
    escrever_execucao(
        runs_dir,
        "run-a",
        [terminal],
        {"run_id": "run-a", "stage": "validacao", "status": status},
    )

    resumo = servico.gerar(runs_dir, tmp_path / "logs")

    assert len(resumo["falhas"]) == 1
    assert resumo["falhas"][0]["error_code"] == terminal["error_code"]
    assert resumo["falhas"][0]["timestamp"] == terminal["timestamp"]
