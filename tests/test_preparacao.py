"""Testes da preparacao e separacao do dataset (issue F1-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento import preparacao as prep

PROPORCOES = {"train": 0.6, "validation": 0.2, "test": 0.1, "simulation_base": 0.1}


def montar_config(tmp_path: Path, origem: Path) -> dict:
    return {
        "contract_version": "1.0",
        "stage": "preparacao",
        "dataset_id": "teste",
        "source": {
            "artifact_type": "origem",
            "path": origem.as_posix(),
            "source_url": None,
            "license": None,
        },
        "target": {
            "column": "loan_status",
            "positive_class": "Charged Off",
            "negative_class": "Fully Paid",
            "positive_value": 1,
            "negative_value": 0,
        },
        "split": {
            "strategy": "temporal",
            "time_column": "issue_d",
            "time_format": "%b-%Y",
            "seed": 42,
            "proportions": PROPORCOES,
        },
        "outputs": {
            "reference_dir": (tmp_path / "reference").as_posix(),
            "analysis_dir": (tmp_path / "analysis").as_posix(),
        },
    }


def gravar_origem(caminho: Path, linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(caminho, index=False, encoding="utf-8")


def origem_sintetica() -> list[dict]:
    meses = ["Jan-2015", "Feb-2015", "Mar-2015", "Apr-2015", "May-2015"]
    linhas = []
    for indice in range(10):
        linhas.append(
            {
                "record_id": f"LC{indice + 1:07d}",
                "issue_d": meses[indice // 2],
                "loan_status": "Charged Off" if indice % 3 == 0 else "Fully Paid",
                "loan_amnt": str(1000 + indice),
            }
        )
    return linhas


def test_calcular_cotas_soma_total() -> None:
    cotas = prep.calcular_cotas(100_000, PROPORCOES)
    assert cotas == {
        "train": 60_000,
        "validation": 20_000,
        "test": 10_000,
        "simulation_base": 10_000,
    }
    assert sum(cotas.values()) == 100_000


def test_calcular_cotas_com_resto() -> None:
    cotas = prep.calcular_cotas(7, PROPORCOES)
    assert sum(cotas.values()) == 7
    assert cotas["train"] >= cotas["validation"] >= cotas["test"]


def test_binarizar_alvo_mapeia_classes() -> None:
    base = pd.DataFrame(
        {
            "record_id": ["LC1", "LC2"],
            "loan_status": ["Fully Paid", "Charged Off"],
        }
    )
    resultado = prep.binarizar_alvo(base, montar_config(Path("."), Path("origem.csv"))["target"])
    assert resultado["loan_status"].tolist() == [0, 1]


def test_binarizar_alvo_rejeita_rotulo_desconhecido() -> None:
    base = pd.DataFrame({"record_id": ["LC1"], "loan_status": ["Current"]})
    with pytest.raises(prep.RejeicaoDados):
        prep.binarizar_alvo(base, montar_config(Path("."), Path("origem.csv"))["target"])


def test_separar_temporal_disjunto_e_ordenado() -> None:
    base = pd.DataFrame(origem_sintetica())
    cotas = prep.calcular_cotas(len(base), PROPORCOES)
    particoes = prep.separar_temporal(base, "issue_d", "%b-%Y", cotas)

    ids = [set(parte["record_id"]) for parte in particoes.values()]
    assert sum(len(conjunto) for conjunto in ids) == len(base)
    assert set().union(*ids) == set(base["record_id"])

    datas = {
        nome: pd.to_datetime(parte["issue_d"], format="%b-%Y") for nome, parte in particoes.items()
    }
    assert datas["train"].max() <= datas["validation"].min()
    assert datas["validation"].max() <= datas["test"].min()
    assert datas["test"].max() <= datas["simulation_base"].min()


def test_processar_sucesso_gera_particoes_e_manifesto(tmp_path: Path) -> None:
    origem = tmp_path / "origem.csv"
    gravar_origem(origem, origem_sintetica())
    config = montar_config(tmp_path, origem)
    config_path = tmp_path / "preparacao.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    output_dir = tmp_path / "runs" / "run-1"
    output_dir.mkdir(parents=True)

    codigo = prep.processar(config_path, config, "run-1", output_dir)

    assert codigo == prep.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["status"] == "success"
    assert manifesto["stage"] == "preparacao"
    assert len(manifesto["outputs"]) == 4
    assert sum(item["row_count"] for item in manifesto["outputs"]) == 10
    for artefato in manifesto["outputs"]:
        assert Path(artefato["path"]).is_file()
        assert artefato["sha256"]
    assert (output_dir / "events.jsonl").is_file()
    assert (output_dir / "metrics.json").is_file()


def test_processar_rejeita_origem_invalida(tmp_path: Path) -> None:
    linhas = origem_sintetica()
    linhas[0]["loan_status"] = "Current"
    origem = tmp_path / "origem.csv"
    gravar_origem(origem, linhas)
    config = montar_config(tmp_path, origem)
    config_path = tmp_path / "preparacao.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    output_dir = tmp_path / "runs" / "run-2"
    output_dir.mkdir(parents=True)

    codigo = prep.processar(config_path, config, "run-2", output_dir)

    assert codigo == prep.CODIGO_REJEICAO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["status"] == "rejected"
    assert manifesto["error_code"] == "REJEICAO_DADOS"


def test_escrever_particoes_nao_sobrescreve(tmp_path: Path) -> None:
    origem = tmp_path / "origem.csv"
    gravar_origem(origem, origem_sintetica())
    config = montar_config(tmp_path, origem)
    base = pd.DataFrame(origem_sintetica())
    cotas = prep.calcular_cotas(len(base), PROPORCOES)
    particoes = prep.separar_temporal(base, "issue_d", "%b-%Y", cotas)

    prep.escrever_particoes(particoes, config, "prep-1")
    with pytest.raises(ValueError):
        prep.escrever_particoes(particoes, config, "prep-1")
