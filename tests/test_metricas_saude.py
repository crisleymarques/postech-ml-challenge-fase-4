"""Testes das metricas de saude operacional do pipeline (issue F3-02)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento import metricas_saude as ms

CATALOGO_REAL = Path(__file__).resolve().parents[1] / "configs" / "metricas_saude.json"


def metrica(**campos: object) -> dict:
    """Metrica minima conforme o catalogo, sobrescrita pelos campos informados."""
    base = {
        "id": "categoria.metrica",
        "categoria": "pipeline",
        "nome": "Metrica",
        "unidade": "unidade",
        "escopo": "execucao",
        "etapas": None,
        "coleta": "fonte",
        "criterio": "criterio",
        "direcao": None,
        "limiar_degracao": None,
        "limiar_falha": None,
    }
    base.update(campos)
    return base


def escrever_run(
    runs_dir: Path,
    run_id: str,
    *,
    manifesto: dict | None = None,
    metrics: dict | None = None,
    quality: dict | None = None,
    drift: dict | None = None,
    analise: dict | None = None,
    performance: dict | None = None,
    eventos: list[dict] | None = None,
) -> Path:
    """Cria um diretorio de execucao com os artefatos informados."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    for nome, conteudo in (
        ("manifest.json", manifesto),
        ("metrics.json", metrics),
        ("quality.json", quality),
        ("drift.json", drift),
        ("analise.json", analise),
        ("performance.json", performance),
    ):
        if conteudo is not None:
            (run_dir / nome).write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")
    if eventos is not None:
        with (run_dir / "events.jsonl").open("w", encoding="utf-8") as arquivo:
            for evento in eventos:
                arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
    return run_dir


def evento_drift(run_id: str, stage: str) -> dict:
    """Evento de alerta de drift conforme o contrato comum."""
    return {
        "timestamp": "2026-11-09T02:50:49.872109Z",
        "run_id": run_id,
        "stage": stage,
        "event": "drift_alert",
        "level": "WARNING",
        "dataset_id": "lending_club_100k",
        "model_id": "baseline-logreg-820dbb61",
        "reference_id": "ref-820dbb61",
        "scenario_id": "data-drift",
        "batch_id": "batch-f201",
        "error_code": None,
    }


def test_carregar_catalogo_real() -> None:
    catalogo = ms.carregar_catalogo(CATALOGO_REAL)

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
        ms.carregar_catalogo(tmp_path / "inexistente.json")


def test_carregar_catalogo_sem_metricas(tmp_path: Path) -> None:
    caminho = tmp_path / "catalogo.json"
    caminho.write_text(json.dumps({"contract_version": "1.0", "metricas": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        ms.carregar_catalogo(caminho)


def test_carregar_catalogo_escopo_invalido(tmp_path: Path) -> None:
    caminho = tmp_path / "catalogo.json"
    catalogo = {
        "contract_version": "1.0",
        "metricas": [metrica(escopo="global")],
    }
    caminho.write_text(json.dumps(catalogo), encoding="utf-8")

    with pytest.raises(ValueError):
        ms.carregar_catalogo(caminho)


def test_carregar_catalogo_id_duplicado(tmp_path: Path) -> None:
    caminho = tmp_path / "catalogo.json"
    catalogo = {
        "contract_version": "1.0",
        "metricas": [metrica(id="a.b"), metrica(id="a.b")],
    }
    caminho.write_text(json.dumps(catalogo), encoding="utf-8")

    with pytest.raises(ValueError):
        ms.carregar_catalogo(caminho)


def test_carregar_catalogo_versao_incompativel(tmp_path: Path) -> None:
    caminho = tmp_path / "catalogo.json"
    catalogo = {"contract_version": "2.0", "metricas": [metrica()]}
    caminho.write_text(json.dumps(catalogo), encoding="utf-8")

    with pytest.raises(ValueError):
        ms.carregar_catalogo(caminho)


def test_metricas_por_escopo() -> None:
    catalogo = {
        "metricas": [
            metrica(id="a", escopo="execucao"),
            metrica(id="b", escopo="agregado"),
            metrica(id="c", escopo="execucao"),
        ]
    }

    execucao = ms.metricas_por_escopo(catalogo, "execucao")
    agregado = ms.metricas_por_escopo(catalogo, "agregado")

    assert [item["id"] for item in execucao] == ["a", "c"]
    assert [item["id"] for item in agregado] == ["b"]


def test_aplicavel() -> None:
    todas = metrica(etapas=None)
    restrita = metrica(etapas=["deteccao", "analise"])

    assert ms.aplicavel(todas, "treinamento") is True
    assert ms.aplicavel(todas, None) is True
    assert ms.aplicavel(restrita, "deteccao") is True
    assert ms.aplicavel(restrita, "treinamento") is False


def test_listar_execucoes_manifesto_ou_eventos(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    escrever_run(runs_dir, "run-b", eventos=[])
    escrever_run(runs_dir, "run-a", manifesto={"run_id": "run-a"})
    (runs_dir / "run-vazio").mkdir()

    execucoes = ms.listar_execucoes(runs_dir)

    assert [caminho.name for caminho in execucoes] == ["run-a", "run-b"]


def test_listar_execucoes_diretorio_ausente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ms.listar_execucoes(tmp_path / "inexistente")


def test_ler_json_tolerante(tmp_path: Path) -> None:
    assert ms.ler_json(tmp_path / "ausente.json") is None

    invalido = tmp_path / "invalido.json"
    invalido.write_text("{nao-eh-json}", encoding="utf-8")
    assert ms.ler_json(invalido) is None


def test_ler_eventos_tolerante(tmp_path: Path) -> None:
    caminho = tmp_path / "events.jsonl"
    caminho.write_text(
        json.dumps(evento_drift("run-a", "deteccao")) + "\n{invalido}\n\n", encoding="utf-8"
    )

    eventos = ms.ler_eventos(caminho)

    assert len(eventos) == 1
    assert eventos[0]["event"] == "drift_alert"


def test_coletar_duracao() -> None:
    manifesto = {
        "started_at": "2026-11-09T02:12:35.456240Z",
        "finished_at": "2026-11-09T02:12:35.971287Z",
    }

    assert ms.coletar_duracao(manifesto) == pytest.approx(0.515047, abs=1e-6)


def test_coletar_duracao_invalida() -> None:
    assert ms.coletar_duracao({}) is None
    assert ms.coletar_duracao({"started_at": "invalido", "finished_at": "x"}) is None


def test_coletar_volume_prefere_saidas() -> None:
    manifesto = {
        "outputs": [{"row_count": 100}],
        "inputs": [{"row_count": 999}],
    }

    assert ms.coletar_volume(manifesto) == 100


def test_coletar_volume_usa_entradas_sem_row_count_nas_saidas() -> None:
    manifesto = {
        "outputs": [{"path": "quality.json"}],
        "inputs": [{"row_count": 361}, {"row_count": 120}],
    }

    assert ms.coletar_volume(manifesto) == 481


def test_coletar_volume_ausente() -> None:
    assert ms.coletar_volume({"outputs": [], "inputs": []}) is None


def test_coletar_violacoes_do_relatorio() -> None:
    quality = {"violations": [{"rule_id": "R_A", "count": 1}, {"rule_id": "R_B", "count": 2}]}

    assert ms.coletar_violacoes(quality, None) == 3


def test_coletar_violacoes_sem_relatorio_usa_metrics() -> None:
    metrics = {
        "measurements": [
            {"name": "linhas_verificadas", "value": 601},
            {"name": "violacoes_total", "value": 7},
        ]
    }

    assert ms.coletar_violacoes(None, metrics) == 7


def test_coletar_violacoes_ausente() -> None:
    assert ms.coletar_violacoes(None, None) is None


def test_coletar_valor_drift_prefere_detectcao() -> None:
    drift = {"summary": {"drift_share": 0.59}}
    analise = {"summary": {"change_share": 0.36}}

    assert ms.coletar_valor_drift(drift, analise, "drift_share", "change_share") == 0.59


def test_coletar_valor_drift_usa_analise_como_fallback() -> None:
    analise = {"summary": {"changed_columns": ["a", "b"]}}

    assert ms.coletar_valor_drift(None, analise, "drifted_columns", "changed_columns") == [
        "a",
        "b",
    ]


def test_coletar_variaveis_drift_conta_lista() -> None:
    drift = {"summary": {"drifted_columns": ["a", "b", "c"]}}

    assert ms.coletar_variaveis_drift(drift, None) == 3


def test_coletar_variaveis_drift_aceita_contagem() -> None:
    drift = {"summary": {"drifted_columns": 13}}

    assert ms.coletar_variaveis_drift(drift, None) == 13


def test_coletar_variaveis_drift_usa_analise() -> None:
    analise = {"summary": {"changed_columns": 8}}

    assert ms.coletar_variaveis_drift(None, analise) == 8


def test_taxa_sucesso_e_contagens() -> None:
    execucoes = [
        {"run_id": "a", "stage": "preparacao", "status": "success"},
        {"run_id": "b", "stage": "validacao", "status": "rejected"},
        {"run_id": "c", "stage": "validacao", "status": "rejected"},
        {"run_id": "d", "stage": "treinamento", "status": "failed"},
    ]

    assert ms.taxa_sucesso(execucoes) == 0.25
    assert ms.contar_por_status(execucoes, "failed") == 1
    assert ms.contar_lotes_rejeitados(execucoes) == 2


def test_taxa_sucesso_sem_execucoes() -> None:
    assert ms.taxa_sucesso([]) is None


def test_contar_alertas() -> None:
    eventos = [evento_drift("a", "deteccao"), {"event": "completed"}, evento_drift("b", "deteccao")]

    assert ms.contar_alertas(eventos) == 2


def test_situacao_numerica_maior_pior() -> None:
    direcao = metrica(
        id="pipeline.duracao_s",
        direcao="maior_pior",
        limiar_degracao=60.0,
        limiar_falha=300.0,
    )

    assert ms.situacao_medida(direcao, 10.0) == ms.SITUACAO_OK
    assert ms.situacao_medida(direcao, 60.0) == ms.SITUACAO_ATENCAO
    assert ms.situacao_medida(direcao, 300.0) == ms.SITUACAO_CRITICO
    assert ms.situacao_medida(direcao, None) == ms.SITUACAO_INDISPONIVEL


def test_situacao_numerica_menor_pior() -> None:
    taxa = metrica(
        id="pipeline.taxa_sucesso",
        direcao="menor_pior",
        limiar_degracao=0.9,
        limiar_falha=0.5,
    )

    assert ms.situacao_medida(taxa, 0.95) == ms.SITUACAO_OK
    assert ms.situacao_medida(taxa, 0.7) == ms.SITUACAO_ATENCAO
    assert ms.situacao_medida(taxa, 0.4) == ms.SITUACAO_CRITICO


def test_situacao_sem_direcao_fica_ok() -> None:
    assert ms.situacao_medida(metrica(valor=None), 5) == ms.SITUACAO_OK


def test_situacao_especiais() -> None:
    status = metrica(id="pipeline.status_execucao")
    assert ms.situacao_medida(status, "success") == ms.SITUACAO_OK
    assert ms.situacao_medida(status, "rejected") == ms.SITUACAO_ATENCAO
    assert ms.situacao_medida(status, "failed") == ms.SITUACAO_CRITICO
    assert ms.situacao_medida(status, "incomplete") == ms.SITUACAO_CRITICO
    assert ms.situacao_medida(status, "outro") == ms.SITUACAO_INDISPONIVEL

    drift = metrica(id="drift.dataset_drift")
    assert ms.situacao_medida(drift, False) == ms.SITUACAO_OK
    assert ms.situacao_medida(drift, True) == ms.SITUACAO_ATENCAO

    escrita = metrica(id="infra.escrita_ok")
    assert ms.situacao_medida(escrita, True) == ms.SITUACAO_OK
    assert ms.situacao_medida(escrita, False) == ms.SITUACAO_CRITICO


def test_medir_execucao_ignora_etapa_nao_aplicavel() -> None:
    restrita = metrica(id="drift.dataset_drift", etapas=["deteccao"])

    assert ms.medir_execucao(restrita, {"stage": "treinamento"}) is None


def test_medir_execucao_coleta_valor() -> None:
    metrica_drift = metrica(id="drift.dataset_drift", etapas=["deteccao"])
    contexto = {"stage": "deteccao", "drift": {"summary": {"dataset_drift": True}}}

    medida = ms.medir_execucao(metrica_drift, contexto)

    assert medida is not None
    assert medida["valor"] is True
    assert medida["disponibilidade"] == ms.DISPONIVEL
    assert medida["situacao"] == ms.SITUACAO_ATENCAO


def test_medir_execucao_indisponivel_sem_artefato() -> None:
    metrica_drift = metrica(id="drift.dataset_drift", etapas=["deteccao"])
    contexto = {"stage": "deteccao", "drift": None}

    medida = ms.medir_execucao(metrica_drift, contexto)

    assert medida is not None
    assert medida["disponibilidade"] == ms.INDISPONIVEL
    assert medida["situacao"] == ms.SITUACAO_INDISPONIVEL


def test_espaco_livre_e_escrita(tmp_path: Path) -> None:
    livre = ms.espaco_livre(tmp_path)

    assert isinstance(livre, float)
    assert livre > 0
    assert ms.escrita_disponivel(tmp_path / "saida") is True
    assert not (tmp_path / "saida" / ".escrita_saude").exists()


def test_gerar_ponta_a_ponta(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    escrever_run(
        runs_dir,
        "run-deteccao",
        manifesto={
            "contract_version": "1.0",
            "run_id": "run-deteccao",
            "stage": "deteccao",
            "status": "success",
            "started_at": "2026-11-09T02:50:49.872109Z",
            "finished_at": "2026-11-09T02:51:05.727987Z",
            "inputs": [{"row_count": 60000}, {"row_count": 10000}],
        },
        drift={
            "summary": {
                "dataset_drift": True,
                "drifted_columns": ["a", "b"],
                "drift_share": 0.5,
            }
        },
        eventos=[
            evento_drift("run-deteccao", "deteccao"),
            {"run_id": "run-deteccao", "stage": "deteccao", "event": "completed"},
        ],
    )
    escrever_run(
        runs_dir,
        "run-validacao",
        manifesto={
            "run_id": "run-validacao",
            "stage": "validacao",
            "status": "rejected",
            "started_at": "2026-11-09T02:12:35.456240Z",
            "finished_at": "2026-11-09T02:12:35.971287Z",
            "inputs": [{"row_count": 601}],
        },
        quality={"approved": False, "violations": [{"rule_id": "R_A", "count": 2}]},
    )

    output_dir = tmp_path / "saude"
    saude = ms.gerar(runs_dir, output_dir, CATALOGO_REAL)

    assert (output_dir / "saude.json").is_file()
    assert saude["resumo"]["execucoes"] == 2

    por_run = {execucao["run_id"]: execucao for execucao in saude["execucoes"]}
    ids_deteccao = {medida["id"] for medida in por_run["run-deteccao"]["metricas"]}
    ids_validacao = {medida["id"] for medida in por_run["run-validacao"]["metricas"]}

    assert "drift.dataset_drift" in ids_deteccao
    assert "qualidade.violacoes_total" not in ids_deteccao
    assert "qualidade.violacoes_total" in ids_validacao
    assert "drift.dataset_drift" not in ids_validacao

    medidas_deteccao = {medida["id"]: medida for medida in por_run["run-deteccao"]["metricas"]}
    assert medidas_deteccao["drift.dataset_drift"]["situacao"] == ms.SITUACAO_ATENCAO
    assert medidas_deteccao["pipeline.status_execucao"]["situacao"] == ms.SITUACAO_OK

    medidas_validacao = {medida["id"]: medida for medida in por_run["run-validacao"]["metricas"]}
    assert medidas_validacao["pipeline.status_execucao"]["situacao"] == ms.SITUACAO_ATENCAO
    assert medidas_validacao["qualidade.violacoes_total"]["valor"] == 2
    assert medidas_validacao["qualidade.violacoes_total"]["situacao"] == ms.SITUACAO_ATENCAO

    agregados = {medida["id"]: medida for medida in saude["agregados"]}
    assert agregados["drift.alertas_total"]["valor"] == 1
    assert agregados["pipeline.falhas_total"]["valor"] == 0
    assert agregados["qualidade.lotes_rejeitados"]["valor"] == 1
    assert agregados["infra.escrita_ok"]["valor"] is True

    assert saude["resumo"]["por_situacao"][ms.SITUACAO_ATENCAO] >= 4


def test_gerar_sem_execucoes(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    saude = ms.gerar(runs_dir, tmp_path / "saude", CATALOGO_REAL)

    assert saude["execucoes"] == []
    assert saude["resumo"]["execucoes"] == 0
    assert saude["resumo"]["medidas"] == len(saude["agregados"])


def test_publicacao_parcial_nao_alimenta_sucesso_ou_metricas_de_drift(tmp_path):
    escrever_run(
        tmp_path / "runs",
        "parcial",
        manifesto={
            "contract_version": "1.0",
            "run_id": "parcial",
            "stage": "deteccao",
            "status": "success",
        },
        drift={"summary": {"dataset_drift": False, "drift_share": 0}},
        eventos=[{"run_id": "parcial", "stage": "deteccao", "event": "started"}],
    )
    saude = ms.gerar(tmp_path / "runs", tmp_path / "saude", CATALOGO_REAL)
    execucao = saude["execucoes"][0]
    assert execucao["status"] == "incomplete"
    medidas = {m["id"]: m for m in execucao["metricas"]}
    assert medidas["pipeline.status_execucao"]["situacao"] == ms.SITUACAO_CRITICO
    assert medidas["drift.dataset_drift"]["disponibilidade"] == ms.INDISPONIVEL
    taxa = next(m for m in saude["agregados"] if m["id"] == "pipeline.taxa_sucesso")
    assert taxa["valor"] == 0


def test_gerar_runs_dir_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ms.gerar(tmp_path / "inexistente", tmp_path / "saude", CATALOGO_REAL)
