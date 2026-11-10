"""Testes do contrato de qualidade de dados (issue F1-03)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from monitoramento import validacao as val

PARTICOES = ("train", "validation", "test")


def montar_regras() -> dict:
    return {
        "identifier": {"column": "record_id", "pattern": "^LC[0-9]{7}$"},
        "target": {"column": "loan_status", "domain": [0, 1]},
        "not_null": ["record_id", "loan_amnt", "term", "loan_status"],
        "unique": ["record_id"],
        "ranges": {"loan_amnt": [500, 40000]},
        "categories": {"term": ["36 months", "60 months"]},
    }


def montar_config(tmp_path: Path) -> dict:
    return {
        "contract_version": "1.0",
        "stage": "validacao",
        "dataset_id": "teste",
        "schema_version": "1.0",
        "artifacts": list(PARTICOES),
        "rules": montar_regras(),
    }


def gerar_particao(quantidade: int, deslocamento: int) -> pd.DataFrame:
    """Cria uma particao valida para o contrato de teste."""
    linhas = []
    for indice in range(quantidade):
        linhas.append(
            {
                "record_id": f"LC{deslocamento + indice:07d}",
                "loan_amnt": str(1000 + (indice % 20) * 500),
                "term": "36 months" if indice % 3 else "60 months",
                "loan_status": str(indice % 2),
            }
        )
    return pd.DataFrame(linhas)


def escrever_particoes(tmp_path: Path, dados: dict[str, pd.DataFrame] | None = None) -> None:
    dados = dados or {
        "train": gerar_particao(20, 0),
        "validation": gerar_particao(10, 1000),
        "test": gerar_particao(10, 2000),
    }
    for nome, base in dados.items():
        caminho = tmp_path / "reference" / "prep-teste" / f"{nome}.csv"
        caminho.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(caminho, index=False, encoding="utf-8")


def escrever_manifesto(tmp_path: Path, config: dict) -> Path:
    artefatos = []
    for nome in PARTICOES:
        caminho = tmp_path / "reference" / "prep-teste" / f"{nome}.csv"
        base = pd.read_csv(caminho, dtype=str, na_filter=False)
        artefatos.append(
            {
                "artifact_type": nome,
                "path": caminho.as_posix(),
                "sha256": val.sha256_arquivo(caminho),
                "producer_run_id": "prep-teste",
                "row_count": int(len(base)),
                "columns": list(base.columns),
            }
        )

    manifesto = {
        "contract_version": "1.0",
        "run_id": "prep-run-1",
        "stage": "preparacao",
        "status": "success",
        "started_at": "2026-01-01T00:00:00.000000Z",
        "finished_at": "2026-01-01T00:00:01.000000Z",
        "config_path": "configs/preparacao.json",
        "inputs": [],
        "outputs": artefatos,
        "dataset_id": config["dataset_id"],
        "target": {
            "column": "loan_status",
            "positive_class": "Charged Off",
            "negative_class": "Fully Paid",
            "positive_value": 1,
            "negative_value": 0,
        },
        "schema_version": None,
        "model_id": None,
        "reference_id": None,
        "scenario_id": None,
        "batch_id": None,
        "error_code": None,
    }
    caminho = tmp_path / "manifest_prep.json"
    caminho.write_text(json.dumps(manifesto), encoding="utf-8")
    return caminho


def escrever_config(tmp_path: Path, config: dict) -> Path:
    caminho = tmp_path / "validacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")
    return caminho


def test_carregar_config_rejeita_poucas_regras(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    config["rules"]["unique"] = []
    config["rules"]["ranges"] = {}
    config["rules"]["categories"] = {}
    config["rules"]["not_null"] = []
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="ao menos"):
        val.carregar_config(caminho)


def test_carregar_config_rejeita_faixa_invertida(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    config["rules"]["ranges"]["loan_amnt"] = [40000, 500]
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="Faixa invalida"):
        val.carregar_config(caminho)


def test_carregar_manifesto_rejeita_dependencia_sem_sucesso(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    caminho = escrever_manifesto(tmp_path, config)
    manifesto = json.loads(caminho.read_text(encoding="utf-8"))
    manifesto["status"] = "rejected"
    caminho.write_text(json.dumps(manifesto), encoding="utf-8")

    with pytest.raises(ValueError, match="Dependencia sem sucesso"):
        val.carregar_manifesto(caminho)


def test_resolver_artefatos_detecta_hash_divergente(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    manifesto["outputs"][0]["sha256"] = "0" * 64

    with pytest.raises(val.ErroArtefato):
        val.resolver_artefatos(manifesto, config)


def test_agregar_violacoes_conta_por_regra() -> None:
    falhas = pd.DataFrame(
        {
            "check": ["R_FAIXA_loan_amnt", "R_FAIXA_loan_amnt", "R_ALVO_DOMINIO"],
            "failure_case": ["999999", "999999", "3"],
        }
    )

    violacoes = val.agregar_violacoes(falhas)

    assert violacoes == [
        {"rule_id": "R_ALVO_DOMINIO", "count": 1},
        {"rule_id": "R_FAIXA_loan_amnt", "count": 2},
    ]


def test_validar_artefato_aprova_dados_validos(tmp_path: Path) -> None:
    escrever_particoes(tmp_path)
    schema = val.construir_contrato(montar_config(tmp_path))
    caminho = tmp_path / "reference" / "prep-teste" / "train.csv"

    linhas, violacoes = val.validar_artefato(caminho, schema)

    assert linhas == 20
    assert violacoes == []


def test_rejeita_texto_em_regras_numericas_e_aceita_nulo_opcional(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    config["rules"]["ranges"]["optional_numeric"] = [0, 100]
    base = gerar_particao(2, 0)
    base["optional_numeric"] = [None, "abc"]
    base.loc[0, "loan_amnt"] = "abc"
    base.loc[1, "loan_status"] = "Charged Off"
    caminho = tmp_path / "numericos.csv"
    base.to_csv(caminho, index=False, encoding="utf-8")

    linhas, violacoes = val.validar_artefato(caminho, val.construir_contrato(config))

    regras = {item["rule_id"]: item["count"] for item in violacoes}
    assert linhas == 2
    assert regras["R_FAIXA_loan_amnt"] == 1
    assert regras["R_FAIXA_optional_numeric"] == 1
    assert regras["R_ALVO_DOMINIO"] == 1
    assert "R_NAO_NULO_optional_numeric" not in regras


def test_perfil_producao_aprova_features_sem_alvo_e_preserva_ids(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    config["profile"] = "production"
    config["artifacts"] = ["features"]
    base = pd.DataFrame(
        {
            "record_id": ["LC0000001", "LC0000002"],
            "loan_amnt": ["1000", "1500"],
            "term": ["36 months", "60 months"],
            "scenario_id": ["controle", "controle"],
            "batch_id": ["lote-1", "lote-1"],
        }
    )
    caminho = tmp_path / "features.csv"
    base.to_csv(caminho, index=False, encoding="utf-8")

    linhas, violacoes = val.validar_artefato(caminho, val.construir_contrato(config))

    assert linhas == 2
    assert violacoes == []
    validado = pd.read_csv(caminho, dtype=str)
    assert validado[["record_id", "scenario_id", "batch_id"]].equals(
        base[["record_id", "scenario_id", "batch_id"]]
    )

    config["artifacts"] = ["features"]
    manifesto = {
        "contract_version": "1.0",
        "run_id": "sim-1",
        "stage": "simulacao",
        "status": "success",
        "dataset_id": config["dataset_id"],
        "scenario_id": "controle",
        "batch_id": "lote-1",
        "outputs": [
            {
                "artifact_type": "features",
                "path": caminho.as_posix(),
                "sha256": val.sha256_arquivo(caminho),
                "producer_run_id": "sim-1",
                "row_count": len(base),
                "columns": list(base.columns),
            }
        ],
    }
    output_dir = tmp_path / "runs" / "production"
    output_dir.mkdir(parents=True)

    codigo = val.processar(
        Path("configs/validacao.json"), config, manifesto, "validation-production", output_dir
    )

    assert codigo == val.CODIGO_SUCESSO
    qualidade = json.loads((output_dir / "quality.json").read_text(encoding="utf-8"))
    assert qualidade["approved"] is True
    assert qualidade["artifacts"][0]["artifact_type"] == "features"
    manifesto_saida = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto_saida["scenario_id"] == "controle"
    assert manifesto_saida["batch_id"] == "lote-1"
    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text().splitlines()
    ]
    assert all(evento["scenario_id"] == "controle" for evento in eventos)
    assert all(evento["batch_id"] == "lote-1" for evento in eventos)


def test_cli_aplica_perfil_producao_e_seleciona_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = montar_config(tmp_path)
    caminho_config = escrever_config(tmp_path, config)
    caminho_dados = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "record_id": ["LC0000001"],
            "loan_amnt": ["1000"],
            "term": ["36 months"],
            "scenario_id": ["controle"],
            "batch_id": ["lote-cli"],
        }
    ).to_csv(caminho_dados, index=False)
    caminho_manifesto = tmp_path / "simulacao.json"
    caminho_manifesto.write_text(
        json.dumps(
            {
                "contract_version": "1.0",
                "run_id": "sim-cli",
                "stage": "simulacao",
                "status": "success",
                "dataset_id": config["dataset_id"],
                "outputs": [
                    {
                        "artifact_type": "features",
                        "path": caminho_dados.as_posix(),
                        "sha256": val.sha256_arquivo(caminho_dados),
                        "producer_run_id": "sim-cli",
                        "row_count": 1,
                        "columns": list(pd.read_csv(caminho_dados, nrows=0).columns),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs" / "cli-production"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validacao.py",
            "--config",
            str(caminho_config),
            "--profile",
            "production",
            "--input-manifest",
            str(caminho_manifesto),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit) as saida:
        val.main()

    assert saida.value.code == val.CODIGO_SUCESSO
    qualidade = json.loads((output_dir / "quality.json").read_text(encoding="utf-8"))
    assert [item["artifact_type"] for item in qualidade["artifacts"]] == ["features"]


def test_validar_artefato_rejeita_anomalias(tmp_path: Path) -> None:
    base = pd.DataFrame(
        [
            {
                "record_id": "LC0000001",
                "loan_amnt": "5000",
                "term": "36 months",
                "loan_status": "0",
            },
            {
                "record_id": "LC0000001",
                "loan_amnt": "999999",
                "term": "72 months",
                "loan_status": "3",
            },
            {"record_id": "XX", "loan_amnt": "", "term": "36 months", "loan_status": "1"},
        ]
    )
    caminho = tmp_path / "train.csv"
    base.to_csv(caminho, index=False, encoding="utf-8")
    schema = val.construir_contrato(montar_config(tmp_path))

    _, violacoes = val.validar_artefato(caminho, schema)

    regras = {item["rule_id"]: item["count"] for item in violacoes}
    assert regras == {
        "R_ALVO_DOMINIO": 1,
        "R_CATEGORIA_term": 1,
        "R_FAIXA_loan_amnt": 1,
        "R_ID_FORMATO": 1,
        "R_NAO_NULO_loan_amnt": 1,
        "R_UNICO_record_id": 2,
    }


def test_processar_sucesso_gera_quality_e_manifesto(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    output_dir = tmp_path / "runs" / "run-1"
    output_dir.mkdir(parents=True)

    codigo = val.processar(Path("configs/validacao.json"), config, manifesto, "run-1", output_dir)

    assert codigo == val.CODIGO_SUCESSO
    quality = json.loads((output_dir / "quality.json").read_text(encoding="utf-8"))
    assert quality["approved"] is True
    assert quality["checked_rows"] == 40
    assert quality["violations"] == []
    assert {item["artifact_type"] for item in quality["artifacts"]} == set(PARTICOES)

    manifesto_saida = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto_saida["status"] == "success"
    assert manifesto_saida["stage"] == "validacao"
    assert manifesto_saida["schema_version"] == "1.0"
    assert manifesto_saida["target_contract"]["positive_class"] == "Charged Off"
    assert manifesto_saida["quality"]["approved"] is True
    assert [item["artifact_type"] for item in manifesto_saida["inputs"]] == list(PARTICOES)
    assert manifesto_saida["outputs"][0]["artifact_type"] == "quality_report"

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert nomes == {"linhas_verificadas", "violacoes_total", "regras_violadas"}
    assert (output_dir / "events.jsonl").is_file()
    assert (output_dir / "config.json").is_file()


def test_processar_rejeita_dados_invalidos(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    invalida = gerar_particao(10, 0)
    invalida.loc[0, "loan_amnt"] = "999999"
    invalida.loc[1, "loan_status"] = "7"
    escrever_particoes(
        tmp_path,
        {
            "train": invalida,
            "validation": gerar_particao(10, 1000),
            "test": gerar_particao(10, 2000),
        },
    )
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    output_dir = tmp_path / "runs" / "run-2"
    output_dir.mkdir(parents=True)

    codigo = val.processar(Path("configs/validacao.json"), config, manifesto, "run-2", output_dir)

    assert codigo == val.CODIGO_REJEICAO
    quality = json.loads((output_dir / "quality.json").read_text(encoding="utf-8"))
    assert quality["approved"] is False
    regras = {item["rule_id"] for item in quality["violations"]}
    assert {"R_FAIXA_loan_amnt", "R_ALVO_DOMINIO"} <= regras

    manifesto_saida = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto_saida["status"] == "rejected"
    assert manifesto_saida["error_code"] == "QUALIDADE_REPROVADA"


def test_processar_falha_com_dataset_incompativel(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    manifesto["dataset_id"] = "outro"
    output_dir = tmp_path / "runs" / "run-3"
    output_dir.mkdir(parents=True)

    codigo = val.processar(Path("configs/validacao.json"), config, manifesto, "run-3", output_dir)

    assert codigo == val.CODIGO_ERRO
    saida = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saida["status"] == "failed"
    assert saida["error_code"] == "ARTEFATO_INCOMPATIVEL"


@pytest.mark.parametrize("metadata", [None, {"column": "loan_status", "positive_value": 1}])
def test_processar_exige_contrato_semantico_do_alvo(
    tmp_path: Path, metadata: dict | None
) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    if metadata is None:
        manifesto.pop("target")
    else:
        manifesto["target"] = metadata
    output_dir = tmp_path / "runs" / "alvo-ausente"
    output_dir.mkdir(parents=True)

    codigo = val.processar(
        Path("configs/validacao.json"), config, manifesto, "alvo-sem-contrato", output_dir
    )

    assert codigo == val.CODIGO_ERRO
    assert not (output_dir / "quality.json").exists()
    resultado = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert resultado["status"] == "failed"
    assert resultado["error_code"] == "ARTEFATO_INCOMPATIVEL"
