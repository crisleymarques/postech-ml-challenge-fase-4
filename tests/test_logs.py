"""Testes da consolidacao dos logs operacionais do pipeline (issue F3-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento import logs as lg


def escrever_execucao(
    runs_dir: Path,
    run_id: str,
    eventos: list[dict],
    manifesto: dict | None = None,
) -> Path:
    """Cria um diretorio de execucao com events.jsonl e manifesto opcional."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as arquivo:
        for evento in eventos:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
    if manifesto is not None:
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as arquivo:
            json.dump(manifesto, arquivo, ensure_ascii=False, indent=2)
    return run_dir


def evento(
    *,
    run_id: str,
    stage: str,
    tipo: str,
    nivel: str = "INFO",
    error_code: str | None = None,
    timestamp: str = "2026-11-09T00:00:00.000000Z",
    scenario_id: str | None = None,
    batch_id: str | None = None,
) -> dict:
    """Evento minimo conforme o contrato comum de observabilidade."""
    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "stage": stage,
        "event": tipo,
        "level": nivel,
        "dataset_id": "teste",
        "model_id": None,
        "reference_id": None,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "error_code": error_code,
    }


def test_categoria_da_etapa_mapeia_etapas_conhecidas() -> None:
    assert lg.categoria_da_etapa("preparacao") == "ingestao"
    assert lg.categoria_da_etapa("validacao") == "validacao"
    assert lg.categoria_da_etapa("deteccao") == "drift"
    assert lg.categoria_da_etapa("analise") == "drift"
    assert lg.categoria_da_etapa("avaliacao") == "avaliacao"


def test_categoria_da_etapa_desconhecida() -> None:
    assert lg.categoria_da_etapa(None) == lg.CATEGORIA_DESCONHECIDA
    assert lg.categoria_da_etapa("etapa-nova") == lg.CATEGORIA_DESCONHECIDA


def test_categorias_da_etapa_preparacao_alimenta_ingestao_e_transformacao() -> None:
    assert lg.categorias_da_etapa("preparacao") == ("ingestao", "transformacao")
    assert lg.categorias_da_etapa("treinamento") == ("treinamento",)


def test_resultado_do_evento() -> None:
    assert lg.resultado_do_evento("started") == "inicio"
    assert lg.resultado_do_evento("completed") == "sucesso"
    assert lg.resultado_do_evento("rejected") == "rejeicao"
    assert lg.resultado_do_evento("failed") == "falha"
    assert lg.resultado_do_evento("skipped") == "ignorado"
    assert lg.resultado_do_evento("drift_alert") == "alerta"
    assert lg.resultado_do_evento("outro") == lg.RESULTADO_DESCONHECIDO


def test_resultado_do_status() -> None:
    assert lg.resultado_do_status("success") == "sucesso"
    assert lg.resultado_do_status("rejected") == "rejeicao"
    assert lg.resultado_do_status("failed") == "falha"
    assert lg.resultado_do_status(None) == "incompleto"


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

    execucoes = lg.listar_execucoes(runs_dir)

    assert [caminho.name for caminho in execucoes] == ["run-a", "run-b"]


def test_listar_execucoes_diretorio_ausente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lg.listar_execucoes(tmp_path / "inexistente")


def test_ler_eventos_ignora_linhas_vazias(tmp_path: Path) -> None:
    caminho = tmp_path / "events.jsonl"
    caminho.write_text(
        json.dumps(evento(run_id="run-a", stage="preparacao", tipo="started")) + "\n\n",
        encoding="utf-8",
    )

    eventos = lg.ler_eventos(caminho)

    assert len(eventos) == 1
    assert eventos[0]["stage"] == "preparacao"


def test_ler_eventos_rejeita_json_invalido(tmp_path: Path) -> None:
    caminho = tmp_path / "events.jsonl"
    caminho.write_text("{nao-eh-json}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        lg.ler_eventos(caminho)


def test_ler_manifesto_ausente_retorna_none(tmp_path: Path) -> None:
    assert lg.ler_manifesto(tmp_path / "manifest.json") is None


def test_ler_manifesto_invalido_retorna_none(tmp_path: Path) -> None:
    caminho = tmp_path / "manifest.json"
    caminho.write_text("{invalido}", encoding="utf-8")

    assert lg.ler_manifesto(caminho) is None


def test_inferir_status_por_evento_terminal() -> None:
    eventos = [
        evento(run_id="run-a", stage="treinamento", tipo="started"),
        evento(run_id="run-a", stage="treinamento", tipo="completed"),
    ]
    assert lg.inferir_status(eventos) == "success"

    rejeitado = [
        evento(run_id="run-a", stage="validacao", tipo="started"),
        evento(run_id="run-a", stage="validacao", tipo="rejected", nivel="ERROR"),
    ]
    assert lg.inferir_status(rejeitado) == "rejected"

    incompleto = [evento(run_id="run-a", stage="treinamento", tipo="started")]
    assert lg.inferir_status(incompleto) == lg.STATUS_INCOMPLETO


def test_classificar_evento_adiciona_categoria_e_resultado() -> None:
    classificado = lg.classificar_evento(
        evento(run_id="run-a", stage="preparacao", tipo="completed")
    )

    assert classificado["categoria"] == "ingestao"
    assert classificado["resultado"] == "sucesso"
    assert classificado["stage"] == "preparacao"


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

    info = lg.analisar_execucao(run_dir, eventos)

    assert info["status"] == "success"
    assert info["stage"] == "preparacao"
    assert info["resultado"] == "sucesso"
    assert info["transformations"] == ["trim_espacos", "binarizar_alvo"]


def test_analisar_execucao_completed_sem_manifesto_e_incompleta(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    eventos = [evento(run_id="run-a", stage="preparacao", tipo="completed")]

    info = lg.analisar_execucao(run_dir, eventos)

    assert info["status"] == "incomplete"
    assert info["stage"] == "preparacao"
    assert info["transformations"] == []


def test_analisar_execucao_sem_manifesto_infere_incompleto(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    eventos = [evento(run_id="run-a", stage="treinamento", tipo="started")]

    info = lg.analisar_execucao(run_dir, eventos)

    assert info["status"] == lg.STATUS_INCOMPLETO
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
    info = lg.analisar_execucao(run_dir, lg.ler_eventos(run_dir / "events.jsonl"))
    assert info["status"] == "incomplete"


def test_manifesto_sem_arquivo_de_eventos_aparece_como_incompleto(tmp_path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {"contract_version": "1.0", "run_id": "run-a", "stage": "deteccao", "status": "success"}
        )
    )
    resumo = lg.gerar(tmp_path, tmp_path / "logs")
    assert resumo["execucoes"]["por_status"] == {"incomplete": 1}


def test_montar_resumo_conta_por_status_e_categoria(tmp_path: Path) -> None:
    eventos = [
        lg.classificar_evento(evento(run_id="run-a", stage="preparacao", tipo="started")),
        lg.classificar_evento(evento(run_id="run-a", stage="preparacao", tipo="completed")),
        lg.classificar_evento(
            evento(
                run_id="run-b",
                stage="validacao",
                tipo="rejected",
                nivel="ERROR",
                error_code="QUALIDADE_REPROVADA",
            )
        ),
        lg.classificar_evento(
            evento(
                run_id="run-c",
                stage="simulacao",
                tipo="drift_alert",
                scenario_id="data-drift",
                batch_id="batch-f201",
            )
        ),
    ]
    execucoes = [
        {
            "run_id": "run-a",
            "stage": "preparacao",
            "status": "success",
            "resultado": "sucesso",
            "transformations": ["trim_espacos"],
            "error_code": None,
        },
        {
            "run_id": "run-b",
            "stage": "validacao",
            "status": "rejected",
            "resultado": "rejeicao",
            "transformations": [],
            "error_code": "QUALIDADE_REPROVADA",
        },
        {
            "run_id": "run-c",
            "stage": "simulacao",
            "status": "success",
            "resultado": "sucesso",
            "transformations": [],
            "error_code": None,
        },
        {
            "run_id": "run-d",
            "stage": "treinamento",
            "status": "incomplete",
            "resultado": "incompleto",
            "transformations": [],
            "error_code": None,
        },
    ]

    resumo = lg.montar_resumo(
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path / "logs",
        execucoes=execucoes,
        eventos=eventos,
    )

    assert resumo["execucoes"]["total"] == 4
    assert resumo["execucoes"]["por_status"] == {"success": 2, "rejected": 1, "incomplete": 1}
    assert resumo["execucoes"]["incompletas"] == ["run-d"]
    assert resumo["eventos"]["total"] == 4
    # A preparacao alimenta ingestao e transformacao.
    assert resumo["por_categoria"]["ingestao"]["inicio"] == 1
    assert resumo["por_categoria"]["transformacao"]["sucesso"] == 1
    assert resumo["por_categoria"]["validacao"]["rejeicao"] == 1
    assert resumo["por_categoria"]["simulacao"]["alerta"] == 1


def test_montar_resumo_reune_falhas_alertas_e_transformacoes(tmp_path: Path) -> None:
    eventos = [
        lg.classificar_evento(
            evento(
                run_id="run-b",
                stage="validacao",
                tipo="rejected",
                nivel="ERROR",
                error_code="QUALIDADE_REPROVADA",
            )
        ),
        lg.classificar_evento(
            evento(
                run_id="run-c",
                stage="simulacao",
                tipo="drift_alert",
                scenario_id="data-drift",
                batch_id="batch-f201",
            )
        ),
    ]
    execucoes = [
        {
            "run_id": "run-a",
            "stage": "preparacao",
            "status": "success",
            "resultado": "sucesso",
            "transformations": ["trim_espacos", "binarizar_alvo"],
            "error_code": None,
        },
        {
            "run_id": "run-b",
            "stage": "validacao",
            "status": "rejected",
            "resultado": "rejeicao",
            "transformations": [],
            "error_code": "QUALIDADE_REPROVADA",
        },
    ]

    execucoes.append(
        {
            "run_id": "run-c",
            "stage": "simulacao",
            "status": "success",
            "resultado": "sucesso",
            "transformations": [],
            "error_code": None,
        }
    )
    resumo = lg.montar_resumo(
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path / "logs",
        execucoes=execucoes,
        eventos=eventos,
    )

    assert len(resumo["falhas"]) == 1
    assert resumo["falhas"][0]["error_code"] == "QUALIDADE_REPROVADA"
    assert resumo["falhas"][0]["run_id"] == "run-b"
    assert len(resumo["alertas"]) == 1
    assert resumo["alertas"][0]["scenario_id"] == "data-drift"
    assert resumo["transformacoes"] == [
        {"run_id": "run-a", "nomes": ["trim_espacos", "binarizar_alvo"], "resultado": "sucesso"}
    ]


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
    resumo = lg.gerar(runs_dir, output_dir)

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

    resumo = lg.gerar(runs_dir, tmp_path / "logs")

    linhas = (
        (tmp_path / "logs" / "operacional.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert json.loads(linhas[0])["run_id"] == "run-a"
    assert json.loads(linhas[1])["run_id"] == "run-b"
    assert resumo["execucoes"]["total"] == 2


def test_gerar_sem_execucoes(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    resumo = lg.gerar(runs_dir, tmp_path / "logs")

    assert resumo["execucoes"]["total"] == 0
    assert resumo["eventos"]["total"] == 0
    assert (tmp_path / "logs" / "operacional.jsonl").read_text(encoding="utf-8") == ""


def test_gerar_runs_dir_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lg.gerar(tmp_path / "inexistente", tmp_path / "logs")


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

    resumo = lg.gerar(runs_dir, tmp_path / "logs")

    assert resumo["execucoes"]["por_status"] == {status: 1}
    assert resumo["falhas"] == [{
        "run_id": "run-a",
        "stage": "validacao",
        "categoria": "validacao",
        "event": status,
        "level": "ERROR",
        "error_code": "QUALIDADE_REPROVADA",
        "timestamp": "2026-11-09T00:00:01.000000Z",
    }]
    # O diagnostico do manifesto nao inventa eventos nem altera a fonte.
    assert resumo["eventos"]["total"] == len(eventos)
    linhas = (tmp_path / "logs" / "operacional.jsonl").read_text().splitlines()
    assert len(linhas) == len(eventos)
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == anteriores


@pytest.mark.parametrize("status", ["failed", "rejected"])
def test_gerar_nao_duplica_falha_presente_nos_eventos(tmp_path: Path, status: str) -> None:
    runs_dir = tmp_path / "runs"
    terminal = evento(
        run_id="run-a", stage="validacao", tipo=status,
        nivel="ERROR", error_code="QUALIDADE_REPROVADA",
    )
    escrever_execucao(
        runs_dir, "run-a", [terminal],
        {"run_id": "run-a", "stage": "validacao", "status": status},
    )

    resumo = lg.gerar(runs_dir, tmp_path / "logs")

    assert len(resumo["falhas"]) == 1
    assert resumo["falhas"][0]["error_code"] == terminal["error_code"]
    assert resumo["falhas"][0]["timestamp"] == terminal["timestamp"]
