"""Testes das funcoes de carga e agregacao do painel de monitoramento (issue F3-04)."""

from __future__ import annotations

import json
from pathlib import Path

from monitoramento import dashboard as dash


def escrever_execucao(
    runs_dir: Path,
    run_id: str,
    *,
    manifesto: dict | None = None,
    eventos: list[dict] | None = None,
) -> Path:
    """Cria um diretorio de execucao com manifesto e/ou events.jsonl."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    if manifesto is not None:
        (run_dir / "manifest.json").write_text(json.dumps(manifesto), encoding="utf-8")
    if eventos is not None:
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as arquivo:
            for evento in eventos:
                arquivo.write(json.dumps(evento) + "\n")
    return run_dir


def evento(run_id: str, stage: str, tipo: str) -> dict:
    """Evento minimo conforme o contrato comum."""
    return {"run_id": run_id, "stage": stage, "event": tipo, "level": "INFO"}


def test_ler_json_ausente(tmp_path: Path) -> None:
    assert dash.ler_json(tmp_path / "inexistente.json") is None


def test_ler_json_invalido(tmp_path: Path) -> None:
    caminho = tmp_path / "saude.json"
    caminho.write_text("{nao e json", encoding="utf-8")
    assert dash.ler_json(caminho) is None


def test_ler_json_lista_retorna_none(tmp_path: Path) -> None:
    caminho = tmp_path / "lista.json"
    caminho.write_text("[1, 2, 3]", encoding="utf-8")
    assert dash.ler_json(caminho) is None


def test_ler_eventos_tolerante(tmp_path: Path) -> None:
    caminho = tmp_path / "events.jsonl"
    caminho.write_text('{"event": "started"}\n\nnao e json\n{"event": "completed"}\n', "utf-8")
    eventos = dash.ler_eventos(caminho)
    assert [evento["event"] for evento in eventos] == ["started", "completed"]


def test_inferir_status_usa_ultimo_terminal() -> None:
    eventos = [
        evento("run-a", "validacao", "started"),
        evento("run-a", "validacao", "completed"),
        evento("run-a", "validacao", "drift_alert"),
    ]
    assert dash.inferir_status(eventos) == dash.STATUS_SUCESSO


def test_inferir_status_sem_terminal_e_incompleto() -> None:
    assert dash.inferir_status([evento("run-a", "validacao", "started")]) == dash.STATUS_INCOMPLETO


def test_listar_execucoes_ignora_diretorio_sem_artefato(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    escrever_execucao(runs_dir, "run-b", manifesto={"run_id": "run-b"})
    escrever_execucao(runs_dir, "run-a", eventos=[evento("run-a", "preparacao", "started")])
    (runs_dir / "vazio").mkdir()
    nomes = [caminho.name for caminho in dash.listar_execucoes(runs_dir)]
    assert nomes == ["run-a", "run-b"]


def test_listar_execucoes_diretorio_inexistente(tmp_path: Path) -> None:
    assert dash.listar_execucoes(tmp_path / "sem-runs") == []


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
    execucoes = dash.carregar_execucoes(runs_dir)

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
    execucoes = dash.carregar_execucoes(runs_dir)

    assert execucoes[0]["status"] == dash.STATUS_INCOMPLETO
    assert execucoes[0]["incompleta"] is True
    assert execucoes[0]["stage"] == "predicao"


def test_resumir_execucoes(tmp_path: Path) -> None:
    execucoes = [
        {"run_id": "a", "stage": "validacao", "status": "success", "incompleta": False},
        {"run_id": "b", "stage": "validacao", "status": "rejected", "incompleta": False},
        {"run_id": "c", "stage": "deteccao", "status": "incomplete", "incompleta": True},
    ]
    resumo = dash.resumir_execucoes(execucoes)

    assert resumo["total"] == 3
    assert resumo["por_status"] == {"success": 1, "rejected": 1, "incomplete": 1}
    assert resumo["por_etapa"] == {"validacao": 2, "deteccao": 1}
    assert resumo["incompletas"] == ["c"]


def test_taxa_sucesso() -> None:
    execucoes = [
        {"status": "success"},
        {"status": "success"},
        {"status": "rejected"},
        {"status": "failed"},
    ]
    assert dash.taxa_sucesso(execucoes) == 0.5
    assert dash.taxa_sucesso([]) is None


def test_carregar_painel_disponibilidade(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    escrever_execucao(runs_dir, "run-a", manifesto={"run_id": "run-a", "status": "success"})
    logs = tmp_path / "resumo.json"
    logs.write_text(json.dumps({"generated_at": "2026-11-09T00:00:00.000000Z"}), encoding="utf-8")

    painel = dash.carregar_painel(runs_dir, logs, tmp_path / "saude.json", tmp_path / "alert.json")

    assert painel["disponibilidade"] == {"logs": True, "saude": False, "alertas": False}
    assert painel["logs"] is not None
    assert painel["saude"] is None
    assert painel["fontes"]["runs_dir"] == runs_dir.as_posix()


def test_indicadores() -> None:
    painel = {
        "execucoes": [{"status": "success"}, {"status": "failed"}],
        "resumo_execucoes": {
            "total": 2,
            "por_status": {"success": 1, "failed": 1},
            "por_etapa": {"treinamento": 1, "deteccao": 1},
            "incompletas": [],
        },
        "alertas": {
            "resumo": {"total": 3, "por_severidade": {"warning": 2, "critical": 1}},
            "drift": {"dataset_drift": True, "variaveis_com_drift": 13},
        },
    }
    ind = dash.indicadores(painel)

    assert ind["execucoes"] == 2
    assert ind["taxa_sucesso"] == 0.5
    assert ind["falhas"] == 1
    assert ind["alertas_total"] == 3
    assert ind["alertas_critical"] == 1
    assert ind["dataset_drift"] is True
    assert ind["variaveis_com_drift"] == 13


def test_indicadores_sem_alertas() -> None:
    painel = {
        "execucoes": [],
        "resumo_execucoes": {
            "total": 0,
            "por_status": {},
            "por_etapa": {},
            "incompletas": [],
        },
        "alertas": None,
    }
    ind = dash.indicadores(painel)

    assert ind["execucoes"] == 0
    assert ind["taxa_sucesso"] is None
    assert ind["alertas_total"] == 0
    assert ind["dataset_drift"] is None


def test_contar_situacoes() -> None:
    saude = {"resumo": {"por_situacao": {"ok": 10, "atencao": 2, "critico": 1}}}
    assert dash.contar_situacoes(saude) == {"ok": 10, "atencao": 2, "critico": 1}
    assert dash.contar_situacoes(None) == {}


def test_medidas_de_saude() -> None:
    saude = {
        "execucoes": [
            {
                "run_id": "run-a",
                "metricas": [
                    {
                        "id": "pipeline.duracao_s",
                        "categoria": "pipeline",
                        "valor": 1.5,
                        "unidade": "segundos",
                        "situacao": "ok",
                    }
                ],
            }
        ],
        "agregados": [
            {
                "id": "drift.alertas_total",
                "categoria": "drift",
                "valor": 3,
                "unidade": "alertas",
                "situacao": "ok",
            }
        ],
    }
    medidas = dash.medidas_de_saude(saude)

    assert len(medidas) == 2
    assert medidas[0]["origem"] == "run-a"
    assert medidas[0]["escopo"] == "execucao"
    assert medidas[1]["origem"] == "agregado"
    assert medidas[1]["escopo"] == "agregado"


def test_medidas_de_saude_ausente() -> None:
    assert dash.medidas_de_saude(None) == []
