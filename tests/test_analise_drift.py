"""Testes da analise estatistica e interpretacao do drift (issue F2-04)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoramento import analise_drift as ad

NUMERICAS = ["annual_inc", "dti", "revol_util"]
CATEGORICAS = ["term", "home_ownership"]
FEATURES = NUMERICAS + CATEGORICAS

SEMENTE = 42
QUANTIDADE = 400


def montar_config() -> dict:
    """Contrato minimo da etapa de analise."""
    return {
        "contract_version": "1.0",
        "stage": "analise",
        "dataset_id": "teste",
        "identifier": "record_id",
        "reference": {
            "producer_stage": "treinamento",
            "features_artifact": "reference_features",
            "metadata_artifact": "model_metadata",
        },
        "batch": {"producer_stage": "simulacao", "features_artifact": "features"},
        "method": {
            "name": "psi_ks",
            "bins": 10,
            "psi_moderate": 0.1,
            "psi_threshold": 0.25,
            "ks_p_value_threshold": 0.05,
            "ks_statistic_threshold": 0.1,
        },
    }


def montar_metadata() -> dict:
    """Metadados compativeis com as features sinteticas."""
    return {
        "contract_version": "1.0",
        "model_id": "modelo-teste",
        "reference_id": "ref-teste",
        "dataset_id": "teste",
        "features": FEATURES,
        "numeric_features": NUMERICAS,
        "categorical_features": CATEGORICAS,
    }


def gerar_features(deslocado: bool, quantidade: int = QUANTIDADE) -> pd.DataFrame:
    """Cria features sinteticas de referencia ou de producao com mudanca clara."""
    rng = np.random.default_rng(SEMENTE + (1 if deslocado else 0))
    if deslocado:
        annual_inc = rng.normal(120000, 5000, quantidade)
        dti = rng.normal(40, 2, quantidade)
        revol_util = rng.normal(80, 5, quantidade)
        term = rng.choice(["36 months", "60 months"], quantidade, p=[0.2, 0.8])
        home = rng.choice(["RENT", "OWN", "MORTGAGE"], quantidade, p=[0.7, 0.2, 0.1])
    else:
        annual_inc = rng.normal(60000, 5000, quantidade)
        dti = rng.normal(10, 2, quantidade)
        revol_util = rng.normal(30, 5, quantidade)
        term = rng.choice(["36 months", "60 months"], quantidade, p=[0.8, 0.2])
        home = rng.choice(["RENT", "OWN", "MORTGAGE"], quantidade, p=[0.2, 0.3, 0.5])

    return pd.DataFrame(
        {
            "record_id": [f"LC{indice:07d}" for indice in range(quantidade)],
            "annual_inc": annual_inc,
            "dti": dti,
            "revol_util": revol_util,
            "term": term,
            "home_ownership": home,
        }
    )


def descritor(tipo: str, caminho: Path, run_id: str) -> dict:
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": ad.sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }


def escrever_manifesto(caminho: Path, **campos: object) -> dict:
    conteudo = {"contract_version": "1.0", "status": "success"}
    conteudo.update(campos)
    caminho.write_text(json.dumps(conteudo), encoding="utf-8")
    return conteudo


def preparar_manifestos(tmp_path: Path) -> list[dict]:
    """Monta os manifestos de treinamento e simulacao com hashes reais."""
    caminho_ref = tmp_path / "reference_features.csv"
    gerar_features(deslocado=False).to_csv(caminho_ref, index=False, encoding="utf-8")
    caminho_lote = tmp_path / "features.csv"
    gerar_features(deslocado=True).to_csv(caminho_lote, index=False, encoding="utf-8")
    caminho_metadata = tmp_path / "metadata.json"
    ad.escrever_json(caminho_metadata, montar_metadata())

    treino = escrever_manifesto(
        tmp_path / "man-treino.json",
        run_id="run-treino",
        stage="treinamento",
        dataset_id="teste",
        model_id="modelo-teste",
        reference_id="ref-teste",
        outputs=[
            descritor("reference_features", caminho_ref, "run-treino"),
            descritor("model_metadata", caminho_metadata, "run-treino"),
        ],
    )
    lote = escrever_manifesto(
        tmp_path / "man-simulacao.json",
        run_id="run-simulacao",
        stage="simulacao",
        dataset_id="teste",
        scenario_id="data-drift",
        batch_id="batch-1",
        schema_version="1.0",
        variaveis_alteradas=["annual_inc", "dti", "revol_util"],
        outputs=[descritor("features", caminho_lote, "run-simulacao")],
    )
    return [treino, lote]


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("method")
    caminho = tmp_path / "analise.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ad.ErroConfiguracao, match="Chaves ausentes"):
        ad.carregar_config(caminho)


def test_carregar_config_rejeita_etapas_iguais(tmp_path: Path) -> None:
    config = montar_config()
    config["batch"]["producer_stage"] = "treinamento"
    caminho = tmp_path / "analise.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ad.ErroConfiguracao, match="devem ser diferentes"):
        ad.carregar_config(caminho)


def test_carregar_config_rejeita_limiar_invalido(tmp_path: Path) -> None:
    config = montar_config()
    config["method"]["psi_threshold"] = 1.5
    caminho = tmp_path / "analise.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ad.ErroConfiguracao, match="entre 0 e 1"):
        ad.carregar_config(caminho)


def test_carregar_config_rejeita_bins_invalido(tmp_path: Path) -> None:
    config = montar_config()
    config["method"]["bins"] = 1
    caminho = tmp_path / "analise.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ad.ErroConfiguracao, match="bins"):
        ad.carregar_config(caminho)


def test_carregar_manifesto_rejeita_status_nao_sucesso(tmp_path: Path) -> None:
    caminho = tmp_path / "manifesto.json"
    caminho.write_text(
        json.dumps({"contract_version": "1.0", "status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(ad.ErroAnalise, match="nao concluiu com sucesso"):
        ad.carregar_manifesto(caminho)


def test_separar_manifestos_rejeita_ausente() -> None:
    manifesto = {"stage": "treinamento", "run_id": "r1"}

    with pytest.raises(ad.ErroAnalise, match="Manifestos ausentes"):
        ad.separar_manifestos([manifesto], montar_config())


def test_separar_manifestos_rejeita_duplicado() -> None:
    duplicados = [
        {"stage": "treinamento", "run_id": "r1"},
        {"stage": "treinamento", "run_id": "r2"},
        {"stage": "simulacao", "run_id": "r3"},
    ]

    with pytest.raises(ad.ErroAnalise, match="duplicado"):
        ad.separar_manifestos(duplicados, montar_config())


def test_verificar_artefato_rejeita_hash_divergente(tmp_path: Path) -> None:
    caminho = tmp_path / "artefato.csv"
    caminho.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ad.ErroAnalise, match="Hash divergente"):
        ad.verificar_artefato({"path": caminho.as_posix(), "sha256": "0" * 64})


def test_ler_features_aplica_tipos(tmp_path: Path) -> None:
    caminho = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "record_id": ["LC1", "LC2"],
            "annual_inc": ["60000.5", ""],
            "term": ["36 months", "60 months"],
        }
    ).to_csv(caminho, index=False, encoding="utf-8")

    base = ad.ler_features(caminho, ["annual_inc"], ["term"])

    assert pd.api.types.is_numeric_dtype(base["annual_inc"])
    assert np.isnan(base.loc[1, "annual_inc"])
    assert base["term"].dtype == object


def test_indice_psi_zero_para_distribuicoes_iguais() -> None:
    igual = np.array([0.5, 0.5])

    assert ad.indice_psi(igual, igual) == pytest.approx(0.0)


def test_indice_psi_positivo_para_distribuicoes_diferentes() -> None:
    esperado = np.array([0.5, 0.5])
    observado = np.array([0.9, 0.1])

    assert ad.indice_psi(esperado, observado) > 0.25


def test_psi_numerico_alto_com_deslocamento() -> None:
    rng = np.random.default_rng(SEMENTE)
    referencia = pd.Series(rng.normal(60000, 5000, QUANTIDADE))
    atual = pd.Series(rng.normal(120000, 5000, QUANTIDADE))

    assert ad.psi_numerico(referencia, atual, 10) > 0.25


def test_psi_categorico_alto_com_proporcoes_diferentes() -> None:
    referencia = pd.Series(["RENT"] * 80 + ["OWN"] * 20)
    atual = pd.Series(["RENT"] * 10 + ["OWN"] * 90)

    assert ad.psi_categorico(referencia, atual) > 0.25


def test_teste_ks_identico_tem_p_valor_alto() -> None:
    rng = np.random.default_rng(SEMENTE)
    amostra = pd.Series(rng.normal(0, 1, QUANTIDADE))

    estatistica, pvalor = ad.teste_ks(amostra, amostra)

    assert estatistica == pytest.approx(0.0)
    assert pvalor > 0.05


def test_teste_ks_detecta_amostras_deslocadas() -> None:
    rng = np.random.default_rng(SEMENTE)
    referencia = pd.Series(rng.normal(0, 1, QUANTIDADE))
    atual = pd.Series(rng.normal(5, 1, QUANTIDADE))

    estatistica, pvalor = ad.teste_ks(referencia, atual)

    assert estatistica > 0.5
    assert pvalor < 0.05


def test_faixa_psi_classifica() -> None:
    assert ad.faixa_psi(0.05, 0.1, 0.25) == "insignificante"
    assert ad.faixa_psi(0.15, 0.1, 0.25) == "moderada"
    assert ad.faixa_psi(0.30, 0.1, 0.25) == "significativa"


def test_analisar_variavel_numerica_detecta_mudanca() -> None:
    rng = np.random.default_rng(SEMENTE)
    referencia = pd.Series(rng.normal(60000, 5000, QUANTIDADE))
    atual = pd.Series(rng.normal(120000, 5000, QUANTIDADE))

    resultado = ad.analisar_variavel(
        "annual_inc",
        ad.TIPO_NUMERICO,
        referencia,
        atual,
        bins=10,
        psi_moderado=0.1,
        psi_limiar=0.25,
        ks_p=0.05,
        ks_limiar=0.1,
    )

    assert resultado["drift_detected"] is True
    assert resultado["psi_band"] == "significativa"
    assert resultado["ks_statistic"] is not None
    assert resultado["reasons"]


def test_analisar_variavel_nao_avaliavel_registra_motivo() -> None:
    vazia = pd.Series([None, None], dtype="float64")

    resultado = ad.analisar_variavel(
        "annual_inc",
        ad.TIPO_NUMERICO,
        vazia,
        vazia,
        bins=10,
        psi_moderado=0.1,
        psi_limiar=0.25,
        ks_p=0.05,
        ks_limiar=0.1,
    )

    assert resultado["drift_detected"] is None
    assert resultado["psi"] is None
    assert resultado["reasons"]


def test_montar_resultados_preserva_ordem() -> None:
    referencia = gerar_features(deslocado=False)
    atual = gerar_features(deslocado=True)

    resultados = ad.montar_resultados(
        referencia,
        atual,
        FEATURES,
        NUMERICAS,
        bins=10,
        psi_moderado=0.1,
        psi_limiar=0.25,
        ks_p=0.05,
        ks_limiar=0.1,
    )

    assert [item["feature"] for item in resultados] == FEATURES
    assert all(item["drift_detected"] is True for item in resultados)


def test_resumir_calcula_proporcao_e_psi_maximo() -> None:
    resultados = [
        {"feature": "a", "drift_detected": True, "psi_band": "significativa", "psi": 0.4},
        {"feature": "b", "drift_detected": False, "psi_band": "insignificante", "psi": 0.02},
        {"feature": "c", "drift_detected": None, "psi_band": None, "psi": None},
    ]

    resumo = ad.resumir(resultados)

    assert resumo["evaluated_columns"] == 2
    assert resumo["changed_columns"] == 1
    assert resumo["change_share"] == 0.5
    assert resumo["significant_columns"] == 1
    assert resumo["psi_maximo"] == 0.4
    assert resumo["changed_features"] == ["a"]


def test_montar_conclusoes_registra_confirmadas_e_ressalva() -> None:
    resumo = {
        "evaluated_columns": 3,
        "changed_columns": 2,
        "significant_columns": 2,
        "changed_features": ["annual_inc", "dti"],
    }

    conclusoes = ad.montar_conclusoes(
        resumo, ["annual_inc", "dti", "revol_util"], psi_limiar=0.25, ks_p=0.05
    )

    texto = " ".join(conclusoes)
    assert "revol_util" in texto
    assert "nao comprova" in texto


def test_processar_publica_analise_e_manifesto(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "analise.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-analise"

    codigo = ad.processar(caminho_config, config, manifestos, "run-analise", output_dir)

    assert codigo == ad.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "analise"
    assert manifesto["status"] == "success"
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["reference_id"] == "ref-teste"
    assert manifesto["scenario_id"] == "data-drift"
    assert manifesto["batch_id"] == "batch-1"

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == {"analise_json"}
    for item in saidas.values():
        assert Path(item["path"]).is_file()
        assert ad.sha256_arquivo(Path(item["path"])) == item["sha256"]

    documento = json.loads((output_dir / "analise.json").read_text(encoding="utf-8"))
    assert documento["tool"]["name"] == "scipy"
    assert documento["method"]["psi_threshold"] == 0.25
    assert documento["method"]["ks_p_value_threshold"] == 0.05
    assert documento["samples"]["reference"]["row_count"] == QUANTIDADE
    assert documento["samples"]["current"]["row_count"] == QUANTIDADE
    assert {item["feature"] for item in documento["features"]} == set(FEATURES)
    assert documento["summary"]["changed_columns"] == len(FEATURES)
    assert set(documento["changed_variables"]["confirmed"]) == {"annual_inc", "dti", "revol_util"}
    assert documento["interpretation"]["conclusions"]

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert "psi_maximo" in nomes
    assert "variaveis_alteradas_confirmadas" in nomes


def test_processar_emite_alerta_de_drift(tmp_path: Path) -> None:
    config = montar_config()
    caminho_config = tmp_path / "analise.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-alerta"

    ad.processar(caminho_config, config, manifestos, "run-alerta", output_dir)

    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "drift_alert", "completed"]
    alerta = next(evento for evento in eventos if evento["event"] == "drift_alert")
    assert alerta["level"] == "WARNING"


def test_processar_rejeita_dataset_divergente(tmp_path: Path) -> None:
    config = montar_config()
    config["dataset_id"] = "outro-dataset"
    caminho_config = tmp_path / "analise.json"
    caminho_config.write_text(json.dumps(config), encoding="utf-8")
    manifestos = preparar_manifestos(tmp_path)
    output_dir = tmp_path / "runs" / "run-analise"

    codigo = ad.processar(caminho_config, config, manifestos, "run-analise", output_dir)

    assert codigo == ad.CODIGO_ERRO
    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert eventos[-1]["event"] == "failed"
    assert eventos[-1]["error_code"] == "ANALISE_INVALIDA"
