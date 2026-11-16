"""ENT-01: servicos reais com origem sintetica, incluindo Evidently e Streamlit."""

import pytest
from streamlit.testing.v1 import AppTest

from monitoramento import integracao
from monitoramento.observabilidade.dashboard import app as painel_app


@pytest.fixture(scope="module")
def demonstracao(tmp_path_factory):
    destino = tmp_path_factory.mktemp("ent01") / "primeira"
    resultado = integracao.executar(destino)
    return destino, resultado


def test_criterios_artefatos_e_bloqueio_antes_da_inferencia(demonstracao):
    destino, resultado = demonstracao
    assert resultado["status"] == "success"
    assert resultado["demonstracao_sintetica"] is True
    assert resultado["predicoes"] == 600
    assert all(
        v is True for k, v in resultado["criterios"].items() if k != "reproducibilidade_confirmada"
    )
    assert resultado["criterios"]["reproducibilidade_confirmada"] is None
    assert len(resultado["etapas"]) == 13
    quality = integracao.ler_json(destino / "runs/rejeicao-reference/quality.json")
    assert {v["rule_id"] for v in quality["violations"]} == {
        "R_ID_FORMATO",
        "R_UNICO_record_id",
        "R_NAO_NULO_loan_amnt",
        "R_FAIXA_loan_amnt",
        "R_CATEGORIA_term",
        "R_ALVO_DOMINIO",
    }
    bloqueio = integracao.ler_json(destino / "runs/predicao-bloqueada/manifest.json")
    assert bloqueio["status"] == "failed"
    assert bloqueio["outputs"] == []
    assert "failed" in (destino / "runs/predicao-bloqueada/events.jsonl").read_text()
    assert set(resultado["variaveis_confirmadas"]) >= {
        "annual_inc",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "revol_util",
    }
    saude = integracao.ler_json(destino / "saude/saude.json")
    medidas = {(e["run_id"], m["id"]): m for e in saude["execucoes"] for m in e["metricas"]}
    assert medidas["predicao", "pipeline.volume_linhas"]["valor"] == 600
    assert medidas["rejeicao-production", "qualidade.violacoes_total"]["valor"] == 1
    drift = integracao.ler_json(destino / "runs/deteccao/drift.json")
    assert (
        medidas["deteccao", "drift.variaveis_com_drift"]["valor"]
        == (drift["summary"]["drifted_columns"])
    )
    assert medidas["deteccao", "drift.proporcao_drift"]["situacao"] == "atencao"


def test_segunda_execucao_reproduz_dados_predicoes_e_estatisticas(demonstracao):
    destino, primeira = demonstracao
    segunda = integracao.executar(
        destino.parent / "segunda", comparar_com=destino / "resultado.json"
    )
    assert segunda["criterios"]["reproducibilidade_confirmada"] is True
    assert primeira["reproducibilidade"] == segunda["reproducibilidade"]


def test_destino_existente_preserva_evidencias(demonstracao):
    destino, _ = demonstracao
    antes = {p: integracao.sha256_arquivo(p) for p in destino.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        integracao.executar(destino)
    assert antes == {p: integracao.sha256_arquivo(p) for p in antes}


def test_falha_na_origem_nao_aprova_integracao(tmp_path, monkeypatch):
    def falhar(*args):
        raise OSError("Falha simulada de persistencia")

    monkeypatch.setattr(integracao, "gerar_origem", falhar)
    destino = tmp_path / "falha"
    with pytest.raises(OSError):
        integracao.executar(destino)
    resultado = integracao.ler_json(destino / "resultado.json")
    assert resultado["status"] == "failed"
    assert "criterios" not in resultado


def pagina():
    from monitoramento.observabilidade.dashboard.app import main

    main()


def test_dashboard_renderiza_as_evidencias_reais_do_fluxo(demonstracao, monkeypatch):
    destino, resultado = demonstracao
    monkeypatch.setattr(painel_app, "RUNS_DIR_PADRAO", destino / "runs")
    monkeypatch.setattr(painel_app, "LOGS_RESUMO_PADRAO", destino / "logs/resumo.json")
    monkeypatch.setattr(painel_app, "SAUDE_PADRAO", destino / "saude/saude.json")
    monkeypatch.setattr(painel_app, "ALERTAS_PADRAO", destino / "alertas/alertas.json")
    app = AppTest.from_function(pagina).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 5
    medidas = {m.label: m.value for m in app.tabs[0].metric}
    assert medidas["Execuções"] == "13"
    assert medidas["Falhas"] == str(resultado["estados"]["failed"])
    assert medidas["Rejeições"] == str(resultado["estados"]["rejected"])
    integrado = integracao.ler_json(destino / "alertas/alertas.json")
    assert medidas["Alertas"] == str(integrado["resumo"]["total"])
    assert not app.tabs[3].dataframe[0].value.empty
    assert not app.tabs[4].dataframe[0].value.empty
