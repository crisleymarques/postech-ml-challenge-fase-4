"""Testes do treinamento e avaliacao do modelo baseline (issue F1-02)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoramento import treinamento as treino
from monitoramento import validacao as val

PARTICOES = ("train", "validation", "test")


def montar_config(tmp_path: Path) -> dict:
    return {
        "contract_version": "1.0",
        "stage": "treinamento",
        "dataset_id": "teste",
        "schema_version": "1.0",
        "model_id": "baseline-teste",
        "seed": 42,
        "target": {
            "column": "loan_status",
            "positive_class": "Charged Off",
            "negative_class": "Fully Paid",
            "positive_value": 1,
            "negative_value": 0,
        },
        "features": {
            "identifier": "record_id",
            "numeric": ["loan_amnt", "annual_inc"],
            "categorical": ["term", "purpose"],
        },
        "model": {
            "algorithm": "logistic_regression",
            "selection_metric": "roc_auc",
            "params": {"C": 1.0, "max_iter": 200, "class_weight": "balanced"},
        },
        "outputs": {
            "models_dir": (tmp_path / "models").as_posix(),
            "reference_dir": (tmp_path / "reference").as_posix(),
        },
    }


def gerar_particao(quantidade: int, semente: int, prefixo: str) -> pd.DataFrame:
    """Cria uma particao sintetica com sinal entre valor, renda e alvo."""
    gerador = np.random.default_rng(semente)
    renda = gerador.integers(20000, 120000, quantidade)
    valor = gerador.integers(1000, 40000, quantidade)
    escore = valor / 40000 - renda / 120000 + gerador.normal(0, 0.3, quantidade)
    alvo = (escore > 0.1).astype(int)

    linhas = []
    for indice in range(quantidade):
        linhas.append(
            {
                "record_id": f"{prefixo}{indice:06d}",
                "loan_amnt": str(int(valor[indice])),
                "annual_inc": str(int(renda[indice])) if indice % 17 else "",
                "term": "36 months" if indice % 3 else "60 months",
                "purpose": ["debt_consolidation", "credit_card", "home_improvement"][indice % 3],
                "loan_status": str(int(alvo[indice])),
            }
        )
    return pd.DataFrame(linhas)


def escrever_particoes(tmp_path: Path) -> dict[str, pd.DataFrame]:
    particoes = {
        "train": gerar_particao(300, 1, "TR"),
        "validation": gerar_particao(120, 2, "VA"),
        "test": gerar_particao(120, 3, "TE"),
    }
    for nome, base in particoes.items():
        caminho = tmp_path / "reference" / "prep-teste" / f"{nome}.csv"
        caminho.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(caminho, index=False, encoding="utf-8")
    return particoes


def escrever_manifesto(tmp_path: Path, config: dict) -> Path:
    artefatos = []
    for nome in PARTICOES:
        caminho = tmp_path / "reference" / "prep-teste" / f"{nome}.csv"
        colunas = pd.read_csv(caminho, nrows=0).columns.tolist()
        base = pd.read_csv(caminho, dtype=str, na_filter=False)
        artefatos.append(
            {
                "artifact_type": nome,
                "path": caminho.as_posix(),
                "sha256": treino.sha256_arquivo(caminho),
                "producer_run_id": "prep-teste",
                "row_count": int(len(base)),
                "columns": colunas,
            }
        )

    manifesto = {
        "contract_version": "1.0",
        "run_id": "validacao-run-1",
        "stage": "validacao",
        "status": "success",
        "started_at": "2026-01-01T00:00:00.000000Z",
        "finished_at": "2026-01-01T00:00:01.000000Z",
        "config_path": "configs/validacao.json",
        "inputs": artefatos,
        "outputs": [],
        "dataset_id": config["dataset_id"],
        "schema_version": config["schema_version"],
        "model_id": None,
        "reference_id": None,
        "scenario_id": None,
        "batch_id": None,
        "error_code": None,
        "quality": {
            "dataset_id": config["dataset_id"],
            "schema_version": config["schema_version"],
            "approved": True,
            "checked_rows": sum(item["row_count"] for item in artefatos),
            "violations": [],
            "artifacts": [
                {
                    "artifact_type": item["artifact_type"],
                    "sha256": item["sha256"],
                    "approved": True,
                }
                for item in artefatos
            ],
        },
    }
    caminho = tmp_path / "manifest_prep.json"
    caminho.write_text(json.dumps(manifesto), encoding="utf-8")
    return caminho


def config_validacao(config: dict) -> dict:
    return {
        "contract_version": "1.0",
        "stage": "validacao",
        "dataset_id": config["dataset_id"],
        "schema_version": config["schema_version"],
        "artifacts": list(PARTICOES),
        "rules": {
            "identifier": {"column": "record_id", "pattern": "^[A-Z]{2}[0-9]{6}$"},
            "target": {"column": "loan_status", "domain": [0, 1]},
            "not_null": ["record_id", "loan_amnt", "term", "loan_status"],
            "unique": ["record_id"],
            "ranges": {"loan_amnt": [500, 40000]},
            "categories": {"term": ["36 months", "60 months"]},
        },
    }


def validar_preparacao(tmp_path: Path, config: dict) -> tuple[int, dict]:
    manifesto_prep = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    manifesto_prep.update(
        {
            "run_id": "prep-run-1",
            "stage": "preparacao",
            "config_path": "configs/preparacao.json",
            "inputs": [],
            "outputs": manifesto_prep["inputs"],
            "schema_version": None,
        }
    )
    manifesto_prep.pop("quality")
    output_dir = tmp_path / "runs" / "validacao"
    output_dir.mkdir(parents=True)
    codigo = val.processar(
        Path("configs/validacao.json"),
        config_validacao(config),
        manifesto_prep,
        "validacao-run-1",
        output_dir,
    )
    return codigo, json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def test_selecionar_features_rejeita_alvo() -> None:
    config = montar_config(Path("."))
    config["features"]["numeric"].append("loan_status")
    base = pd.DataFrame(columns=["record_id", "loan_status", "loan_amnt", "annual_inc", "term"])

    with pytest.raises(treino.ErroArtefato):
        treino.selecionar_features(config, base)


def test_selecionar_features_rejeita_coluna_ausente() -> None:
    config = montar_config(Path("."))
    base = pd.DataFrame(columns=["record_id", "loan_amnt", "term", "purpose"])

    with pytest.raises(treino.ErroArtefato):
        treino.selecionar_features(config, base)


def test_alvo_numerico_rejeita_rotulo_inesperado() -> None:
    config = montar_config(Path("."))
    base = pd.DataFrame({"loan_status": ["0", "1", "2"]})

    with pytest.raises(treino.ErroArtefato):
        treino.alvo_numerico(base, config)


def test_pipeline_ajusta_e_prediz_probabilidades() -> None:
    config = montar_config(Path("."))
    base = gerar_particao(200, 7, "TR")
    features = ["loan_amnt", "annual_inc", "term", "purpose"]
    pipeline = treino.construir_pipeline(["loan_amnt", "annual_inc"], ["term", "purpose"], config)

    alvo = treino.alvo_numerico(base, config)
    pipeline.fit(base[features], alvo)
    probabilidade = pipeline.predict_proba(base[features])[:, 1]

    assert len(probabilidade) == len(alvo)
    assert probabilidade.min() >= 0.0
    assert probabilidade.max() <= 1.0


def test_avaliar_registra_metricas(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    base = gerar_particao(200, 11, "TR")
    features = ["loan_amnt", "annual_inc", "term", "purpose"]
    pipeline = treino.construir_pipeline(["loan_amnt", "annual_inc"], ["term", "purpose"], config)
    pipeline.fit(base[features], treino.alvo_numerico(base, config))

    valores = treino.avaliar(pipeline, base, features, config)

    assert set(valores) == set(treino.METRICAS)
    assert 0.0 <= valores["accuracy"] <= 1.0
    assert valores["roc_auc"] is not None and valores["roc_auc"] > 0.5


def test_calcular_assinatura_estavel_e_sensivel() -> None:
    config = montar_config(Path("."))
    numericas = ["loan_amnt", "annual_inc"]
    categoricas = ["term", "purpose"]

    primeira = treino.calcular_assinatura(config, numericas, categoricas, "abc")
    assert primeira == treino.calcular_assinatura(config, numericas, categoricas, "abc")

    config_alterada = montar_config(Path("."))
    config_alterada["seed"] = 7
    assert primeira != treino.calcular_assinatura(config_alterada, numericas, categoricas, "abc")


def test_resolver_particoes_detecta_hash_divergente(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    manifesto["inputs"][0]["sha256"] = "0" * 64

    with pytest.raises(treino.ErroArtefato):
        treino.resolver_particoes(manifesto)


def test_carregar_manifesto_rejeita_dependencia_sem_sucesso(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    caminho = escrever_manifesto(tmp_path, config)
    manifesto = json.loads(caminho.read_text(encoding="utf-8"))
    manifesto["status"] = "rejected"
    caminho.write_text(json.dumps(manifesto), encoding="utf-8")

    with pytest.raises(ValueError, match="Dependencia sem sucesso"):
        treino.carregar_manifesto(caminho)


def test_processar_sucesso_gera_modelo_avaliacao_e_referencia(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    output_dir = tmp_path / "runs" / "run-1"
    output_dir.mkdir(parents=True)

    codigo = treino.processar(
        Path("configs/treinamento.json"), config, manifesto, "run-1", output_dir
    )

    assert codigo == treino.CODIGO_SUCESSO
    manifesto_saida = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto_saida["status"] == "success"
    assert manifesto_saida["stage"] == "treinamento"
    assert manifesto_saida["model_id"]
    assert manifesto_saida["reference_id"]
    assert manifesto_saida["selection"]["partition"] == "validation"

    diretorio_modelo = Path(config["outputs"]["models_dir"]) / manifesto_saida["model_id"]
    assert (diretorio_modelo / "pipeline.pkl").is_file()
    metadados = json.loads((diretorio_modelo / "metadata.json").read_text(encoding="utf-8"))
    assert metadados["features"] == ["loan_amnt", "annual_inc", "term", "purpose"]
    assert metadados["target"]["positive_class"] == "Charged Off"
    assert metadados["environment"]["scikit_learn"]

    avaliacao = json.loads((diretorio_modelo / "evaluation.json").read_text(encoding="utf-8"))
    particoes = {item["partition"] for item in avaliacao["evaluations"]}
    assert particoes == set(PARTICOES)
    assert len(avaliacao["evaluations"]) == len(PARTICOES) * len(treino.METRICAS)
    assert all(item["valid_labels"] > 0 for item in avaliacao["evaluations"])

    referencia = Path(config["outputs"]["reference_dir"]) / manifesto_saida["reference_id"]
    congelada = pd.read_csv(referencia / "features.csv", dtype=str, na_filter=False)
    assert list(congelada.columns) == ["record_id", "loan_amnt", "annual_inc", "term", "purpose"]
    assert len(congelada) == 300

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {item["name"] for item in metricas["measurements"]} == {
        "roc_auc_validation",
        "roc_auc_test",
    }
    assert (output_dir / "events.jsonl").is_file()
    assert (output_dir / "config.json").is_file()


def test_processar_falha_com_dataset_incompativel(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    manifesto["dataset_id"] = "outro"
    output_dir = tmp_path / "runs" / "run-2"
    output_dir.mkdir(parents=True)

    codigo = treino.processar(
        Path("configs/treinamento.json"), config, manifesto, "run-2", output_dir
    )

    assert codigo == treino.CODIGO_ERRO
    saida = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saida["status"] == "failed"
    assert saida["error_code"] == "ARTEFATO_INCOMPATIVEL"


def test_fluxo_preparacao_validacao_treinamento_aprovado(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    codigo_validacao, manifesto_validacao = validar_preparacao(tmp_path, config)
    assert codigo_validacao == val.CODIGO_SUCESSO

    output_dir = tmp_path / "runs" / "treinamento"
    output_dir.mkdir()
    codigo = treino.processar(
        Path("configs/treinamento.json"), config, manifesto_validacao, "treino-ok", output_dir
    )

    assert codigo == treino.CODIGO_SUCESSO
    manifesto_treino = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metadados = json.loads(
        (Path(config["outputs"]["models_dir"]) / manifesto_treino["model_id"] / "metadata.json")
        .read_text(encoding="utf-8")
    )
    assert metadados["schema_version"] == config["schema_version"]
    assert metadados["quality_validation_run_id"] == "validacao-run-1"


def test_fluxo_validacao_rejeitada_bloqueia_treinamento(tmp_path: Path) -> None:
    config = montar_config(tmp_path)
    particoes = escrever_particoes(tmp_path)
    particoes["train"].loc[0, "loan_amnt"] = "not-a-number"
    particoes["train"].to_csv(
        tmp_path / "reference" / "prep-teste" / "train.csv", index=False, encoding="utf-8"
    )
    codigo_validacao, manifesto_validacao = validar_preparacao(tmp_path, config)
    assert codigo_validacao == val.CODIGO_REJEICAO

    output_dir = tmp_path / "runs" / "treinamento-rejeitado"
    output_dir.mkdir()
    codigo = treino.processar(
        Path("configs/treinamento.json"),
        config,
        manifesto_validacao,
        "treino-rejeitado",
        output_dir,
    )

    assert codigo == treino.CODIGO_ERRO
    assert not list(Path(config["outputs"]["models_dir"]).glob("*"))


@pytest.mark.parametrize(
    "alterar",
    [
        "preparacao_direta",
        "rejected",
        "hash",
        "schema_version",
        "contract_version",
        "effective_schema",
    ],
)
def test_barreira_bloqueia_manifiestos_sem_aprovacao_vinculada(
    tmp_path: Path, alterar: str
) -> None:
    config = montar_config(tmp_path)
    escrever_particoes(tmp_path)
    manifesto = json.loads(escrever_manifesto(tmp_path, config).read_text(encoding="utf-8"))
    if alterar == "preparacao_direta":
        manifesto["stage"] = "preparacao"
    elif alterar == "rejected":
        manifesto["status"] = "rejected"
        manifesto["quality"]["approved"] = False
    elif alterar == "hash":
        caminho = Path(manifesto["inputs"][0]["path"])
        caminho.write_bytes(caminho.read_bytes() + b"\n")
    elif alterar == "schema_version":
        manifesto["schema_version"] = "2.0"
    elif alterar == "contract_version":
        manifesto["contract_version"] = "2.0"
    else:
        manifesto["inputs"][0]["columns"].remove("record_id")
    output_dir = tmp_path / "runs" / f"invalid-{alterar}"
    output_dir.mkdir(parents=True)

    codigo = treino.processar(
        Path("configs/treinamento.json"), config, manifesto, "treino-bloqueado", output_dir
    )

    assert codigo == treino.CODIGO_ERRO
    assert not list(Path(config["outputs"]["models_dir"]).glob("*"))
