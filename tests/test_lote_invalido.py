"""Testes do gerador de lote invalido e do bloqueio de ingestao (issue F1-04)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento import lote_invalido as li
from monitoramento import treinamento as treino
from monitoramento import validacao as val


def montar_config() -> dict:
    """Replica um contrato minimo compativel com as anomalias previstas."""
    return {
        "contract_version": "1.0",
        "stage": "validacao",
        "dataset_id": "teste",
        "schema_version": "1.0",
        "profile": "reference",
        "artifacts": list(li.PARTICOES),
        "target_contract": {
            "column": "loan_status",
            "positive_class": "Charged Off",
            "negative_class": "Fully Paid",
            "positive_value": 1,
            "negative_value": 0,
        },
        "rules": {
            "identifier": {"column": "record_id", "pattern": "^LC[0-9]{7}$"},
            "target": {"column": "loan_status", "domain": [0, 1]},
            "not_null": ["record_id", "loan_amnt", "term", "loan_status"],
            "unique": ["record_id"],
            "ranges": {"loan_amnt": [500, 40000]},
            "categories": {"term": ["36 months", "60 months"]},
        },
    }


def gerar_origem(quantidade: int = 60) -> pd.DataFrame:
    """Cria uma origem valida e unica para servir de base ao lote."""
    linhas = []
    for indice in range(quantidade):
        linhas.append(
            {
                "record_id": f"LC{indice + 1:07d}",
                "loan_amnt": str(1000 + (indice % 20) * 500),
                "term": "36 months" if indice % 3 else "60 months",
                "loan_status": str(indice % 2),
            }
        )
    return pd.DataFrame(linhas)


def escrever_config(tmp_path: Path, config: dict) -> Path:
    caminho = tmp_path / "validacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")
    return caminho


def escrever_origem(tmp_path: Path, quantidade: int = 60) -> Path:
    caminho = tmp_path / "origem.csv"
    gerar_origem(quantidade).to_csv(caminho, index=False, encoding="utf-8")
    return caminho


def test_carregar_config_rejeita_sem_artifacts(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("artifacts")
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="Chaves ausentes"):
        li.carregar_config(caminho)


def test_carregar_config_rejeita_versao_incompativel(tmp_path: Path) -> None:
    config = montar_config()
    config["contract_version"] = "2.0"
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="incompativel"):
        li.carregar_config(caminho)


@pytest.mark.parametrize(
    "profile,artifacts",
    [
        ("production", list(li.PARTICOES)),
        ("production", ["features"]),
        ("reference", ["train", "validation", "features"]),
        ("reference", ["train", "validation", "unknown"]),
        ("reference", ["train", "train", "test"]),
    ],
)
def test_carregar_config_rejeita_perfil_ou_artifacts_nao_suportados(
    tmp_path: Path, profile: str, artifacts: list[str]
) -> None:
    config = montar_config()
    config["profile"] = profile
    config["artifacts"] = artifacts
    config_path = escrever_config(tmp_path, config)
    with pytest.raises(ValueError):
        li.carregar_config(config_path)
    output_dir = tmp_path / "run"
    assert (
        li.gerar(
            config_path=config_path,
            source_path=tmp_path / "origem.csv",
            analysis_dir=tmp_path / "analysis",
            batch_id="b",
            run_id="r",
            output_dir=output_dir,
            total=30,
            seed=42,
        )
        == li.CODIGO_ERRO
    )
    assert not output_dir.exists()


def test_carregar_base_rejeita_origem_ausente(tmp_path: Path) -> None:
    with pytest.raises(li.ErroLote, match="nao encontrada"):
        li.carregar_base(tmp_path / "ausente.csv")


def test_dividir_lote_respeita_tamanhos_e_sem_sobreposicao() -> None:
    base = gerar_origem(60)

    fatias = li.dividir_lote(base, li.PARTICOES, 30, 42)

    assert {nome: len(parte) for nome, parte in fatias.items()} == {
        "train": 18,
        "validation": 6,
        "test": 6,
    }
    combinado = pd.concat(fatias.values(), ignore_index=True)
    assert len(combinado) == 30
    assert combinado["record_id"].is_unique


def test_dividir_lote_rejeita_base_insuficiente() -> None:
    base = gerar_origem(10)

    with pytest.raises(li.ErroLote, match="insuficiente"):
        li.dividir_lote(base, li.PARTICOES, 30, 42)


def test_aplicar_anomalias_injeta_valor_e_duplicata() -> None:
    base = gerar_origem(4)

    ajustado, descritas = li.aplicar_anomalias(base, li.ANOMALIAS["train"])

    assert len(ajustado) == len(base) + 1
    assert ajustado.at[0, "record_id"] == "LC12345"
    assert ajustado["record_id"].duplicated().any()
    assert [item["rule_id"] for item in descritas] == ["R_ID_FORMATO", "R_UNICO_record_id"]


def test_aplicar_anomalias_rejeita_coluna_ausente() -> None:
    base = pd.DataFrame({"record_id": ["LC0000001", "LC0000002"]})

    with pytest.raises(li.ErroLote, match="Coluna da anomalia ausente"):
        li.aplicar_anomalias(base, li.ANOMALIAS["validation"])


def test_gerar_publica_manifesto_e_artefatos(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)
    output_dir = tmp_path / "runs" / "run-lote"

    codigo = li.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="batch-1",
        run_id="run-lote",
        output_dir=output_dir,
        total=30,
        seed=42,
    )

    assert codigo == li.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "simulacao"
    assert manifesto["status"] == "success"
    assert manifesto["scenario_id"] == "lote-invalido"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["seed"] == 42
    assert len(manifesto["anomalies"]) == 6
    assert set(manifesto["variaveis_alteradas"]) == {
        "loan_amnt",
        "loan_status",
        "record_id",
        "term",
    }

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == set(li.PARTICOES)
    assert saidas["train"]["row_count"] == 19
    assert saidas["validation"]["row_count"] == 6
    assert saidas["test"]["row_count"] == 6
    for item in saidas.values():
        assert Path(item["path"]).is_file()
        assert li.sha256_arquivo(Path(item["path"])) == item["sha256"]

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert nomes == {"linhas_geradas", "anomalias_total", "regras_exercitadas"}
    assert (output_dir / "events.jsonl").is_file()
    assert (output_dir / "config.json").is_file()
    assert manifesto["target"] == montar_config()["target_contract"]


def test_gerar_nao_sobrescreve_execucao_ou_lote_existente(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)
    output_dir = tmp_path / "runs" / "run-lote"
    kwargs = dict(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="batch-1",
        run_id="run-lote",
        output_dir=output_dir,
        total=30,
        seed=42,
    )
    assert li.gerar(**kwargs) == li.CODIGO_SUCESSO
    manifest_hash = li.sha256_arquivo(output_dir / "manifest.json")
    csv_hashes = {
        path.name: li.sha256_arquivo(path)
        for path in (tmp_path / "analysis" / "batch-1").glob("*.csv")
    }
    kwargs["seed"] = 7
    assert li.gerar(**kwargs) == li.CODIGO_ERRO
    kwargs["output_dir"] = tmp_path / "runs" / "run-novo"
    kwargs["run_id"] = "run-novo"
    assert li.gerar(**kwargs) == li.CODIGO_ERRO
    assert li.sha256_arquivo(output_dir / "manifest.json") == manifest_hash
    assert {
        path.name: li.sha256_arquivo(path)
        for path in (tmp_path / "analysis" / "batch-1").glob("*.csv")
    } == csv_hashes


@pytest.mark.parametrize("falha", ["origem_ausente", "csv_malformado", "lote_insuficiente"])
def test_erro_apos_inicio_publica_manifesto_failed(tmp_path: Path, falha: str) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    output_dir = tmp_path / "runs" / "run-falha"
    origem = tmp_path / "ausente.csv"
    if falha == "csv_malformado":
        origem.write_text('record_id,loan_amnt\n"sem fechamento', encoding="utf-8")
    elif falha == "lote_insuficiente":
        origem = escrever_origem(tmp_path, quantidade=10)
    codigo = li.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="b",
        run_id="r",
        output_dir=output_dir,
        total=30,
        seed=42,
    )
    assert codigo == li.CODIGO_ERRO
    manifesto = json.loads((output_dir / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["error_code"] == "ERRO_ENTRADA"
    assert manifesto["inputs"] == []
    assert manifesto["outputs"] == []
    assert json.loads((output_dir / "metrics.json").read_text())["measurements"] == []
    assert Path(manifesto["config_path"]).is_file()
    assert [
        json.loads(linha)["event"]
        for linha in (output_dir / "events.jsonl").read_text().splitlines()
    ] == ["started", "failed"]


def test_falha_na_publicacao_nao_registra_sucesso(tmp_path: Path, monkeypatch) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)
    output_dir = tmp_path / "run"
    escrever = li.escrever_json
    gravacoes = []
    falhou = False

    def gravar(caminho, conteudo):
        nonlocal falhou
        if caminho.name == "metrics.json" and not falhou:
            falhou = True
            raise OSError("Falha simulada na publicacao")
        escrever(caminho, conteudo)
        gravacoes.append(caminho.name)

    monkeypatch.setattr(li, "escrever_json", gravar)
    assert (
        li.gerar(
            config_path=config_path,
            source_path=origem,
            analysis_dir=tmp_path / "analysis",
            batch_id="b",
            run_id="r",
            output_dir=output_dir,
            total=30,
            seed=42,
        )
        == li.CODIGO_ERRO
    )
    manifesto = json.loads((output_dir / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["error_code"] == "ERRO_TECNICO"
    assert gravacoes[-1] == "manifest.json"
    assert [
        json.loads(linha)["event"]
        for linha in (output_dir / "events.jsonl").read_text().splitlines()
    ] == ["started", "failed"]


def test_lote_invalido_e_reprovado_pela_validacao(tmp_path: Path) -> None:
    config = montar_config()
    config_path = escrever_config(tmp_path, config)
    origem = escrever_origem(tmp_path)
    lote_dir = tmp_path / "runs" / "run-lote"

    assert (
        li.gerar(
            config_path=config_path,
            source_path=origem,
            analysis_dir=tmp_path / "analysis",
            batch_id="batch-1",
            run_id="run-lote",
            output_dir=lote_dir,
            total=30,
            seed=42,
        )
        == li.CODIGO_SUCESSO
    )

    manifesto = val.carregar_manifesto(lote_dir / "manifest.json")
    valid_dir = tmp_path / "runs" / "run-valid"
    valid_dir.mkdir(parents=True)

    codigo = val.processar(config_path, config, manifesto, "run-valid", valid_dir)

    assert codigo == val.CODIGO_REJEICAO
    quality = json.loads((valid_dir / "quality.json").read_text(encoding="utf-8"))
    assert quality["approved"] is False
    assert {item["rule_id"] for item in quality["violations"]} == {
        "R_ALVO_DOMINIO",
        "R_CATEGORIA_term",
        "R_FAIXA_loan_amnt",
        "R_ID_FORMATO",
        "R_NAO_NULO_loan_amnt",
        "R_UNICO_record_id",
    }

    saida = json.loads((valid_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saida["status"] == "rejected"
    assert saida["error_code"] == "QUALIDADE_REPROVADA"
    with pytest.raises(ValueError, match="Dependencia sem sucesso"):
        treino.carregar_manifesto(valid_dir / "manifest.json")
    assert not (tmp_path / "models").exists()
