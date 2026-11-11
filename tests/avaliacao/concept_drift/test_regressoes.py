"""Regressoes da barreira de qualidade, dos rotulos e da publicacao da avaliacao."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pandas as pd
import pytest

from monitoramento.avaliacao.concept_drift import entradas, servico
from monitoramento.avaliacao.concept_drift.config import ErroAvaliacao
from monitoramento.infraestrutura.arquivos import sha256_arquivo

from .apoio import montar_config, preparar_manifestos


@pytest.mark.parametrize(
    "defeito",
    [
        "ausente",
        "rejeitada",
        "nao_aprovada",
        "batch_id",
        "schema_version",
        "dataset_id",
        "hash",
        "produtor",
        "arquivo_alterado",
        "sem_hash",
        "modelo_schema",
    ],
)
def test_bloqueia_inferencia_sem_aprovacao_compativel(tmp_path: Path, monkeypatch, defeito):
    manifestos = preparar_manifestos(tmp_path)
    modelo, lote, validacao = manifestos
    if defeito == "ausente":
        manifestos.pop()
    elif defeito == "rejeitada":
        validacao["status"] = "rejected"
    elif defeito == "nao_aprovada":
        validacao["quality"]["approved"] = False
    elif defeito in ("batch_id", "schema_version", "dataset_id"):
        validacao[defeito] = "outro"
    elif defeito == "hash":
        validacao["quality"]["artifacts"][0]["sha256"] = "0" * 64
    elif defeito == "produtor":
        validacao["inputs"][0]["producer_run_id"] = "outro"
    elif defeito == "arquivo_alterado":
        features = Path(lote["outputs"][0]["path"])
        base = pd.read_csv(features)
        base["annual_inc"] = -1
        base.to_csv(features, index=False)
    elif defeito == "sem_hash":
        modelo["outputs"][0].pop("sha256")
    elif defeito == "modelo_schema":
        modelo["schema_version"] = "2.0"
    chamadas = []
    monkeypatch.setattr(servico, "prever", lambda *_args: chamadas.append(True))
    output = tmp_path / "run"
    assert (
        servico.processar(tmp_path / "config.json", montar_config(), manifestos, "r", output) == 1
    )
    assert chamadas == []
    falha = json.loads((output / "manifest.json").read_text())
    assert falha["status"] == "failed" and falha["outputs"] == []
    assert json.loads((output / "events.jsonl").read_text().splitlines()[-1])["event"] == "failed"


@pytest.mark.parametrize("rotulo", ["0.9", "1.9", "-1", "2", "inf", "NaN", ""])
def test_alinhamento_rejeita_rotulos_invalidos(tmp_path: Path, rotulo: str) -> None:
    caminho = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": ["a", "b"], "y": [rotulo, "1"]}).to_csv(caminho, index=False)
    with pytest.raises(ErroAvaliacao):
        entradas.juntar_rotulos(pd.Series(["a", "b"]), caminho, "y")


@pytest.mark.parametrize("ids", [["a", "a"], ["a", "b", "extra"]])
def test_rotulos_rejeitam_ids_duplicados_ou_extras(tmp_path: Path, ids: list[str]) -> None:
    caminho = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": ids, "y": [0] * len(ids)}).to_csv(caminho, index=False)
    with pytest.raises(ErroAvaliacao):
        entradas.juntar_rotulos(pd.Series(["a", "b"]), caminho, "y")


def test_auc_indisponivel_publica_estado_nao_avaliavel(tmp_path: Path, capsys) -> None:
    manifestos = preparar_manifestos(tmp_path)
    for item in manifestos[1]["outputs"]:
        if item["artifact_type"] in ("labels", "labels_reference"):
            base = pd.read_csv(item["path"])
            base["loan_status"] = 0
            base.to_csv(item["path"], index=False)
            item["sha256"] = sha256_arquivo(Path(item["path"]))
    output = tmp_path / "run"
    assert (
        servico.processar(tmp_path / "config.json", montar_config(), manifestos, "r", output) == 0
    )
    documento = json.loads((output / "performance.json").read_text())
    assert documento["comparison"]["degradation_detected"] is None
    assert documento["comparison"]["availability"] == "not_available"
    assert json.loads((output / "manifest.json").read_text())["status"] == "success"
    eventos = (output / "events.jsonl").read_text()
    assert "drift_alert" not in eventos
    medidas = json.loads((output / "metrics.json").read_text())["measurements"]
    assert all(m["value"] is None and m["availability"] == "not_available" for m in medidas)
    assert "indisponivel" in capsys.readouterr().out


def test_reexecucao_preserva_avaliacao(tmp_path: Path) -> None:
    manifestos = preparar_manifestos(tmp_path)
    config = montar_config()
    output = tmp_path / "run"
    assert servico.processar(tmp_path / "config.json", config, manifestos, "r", output) == 0
    antes = {p.name: p.read_bytes() for p in output.iterdir()}
    assert servico.processar(tmp_path / "config.json", config, [], "r", output) == 1
    assert {p.name: p.read_bytes() for p in output.iterdir()} == antes


def test_concorrencia_reserva_avaliacao_uma_vez(tmp_path: Path) -> None:
    manifestos = preparar_manifestos(tmp_path)
    barreira = Barrier(2)
    output = tmp_path / "run"

    def executar(_):
        barreira.wait(timeout=10)
        return servico.processar(tmp_path / "config.json", montar_config(), manifestos, "r", output)

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(executar, [0, 1])) == [0, 1]
    assert json.loads((output / "manifest.json").read_text())["status"] == "success"


@pytest.mark.parametrize("arquivo", ["performance.json", "manifest.json"])
def test_falha_publicacao_nao_deixa_sucesso(tmp_path: Path, monkeypatch, arquivo: str) -> None:
    manifestos = preparar_manifestos(tmp_path)
    original = servico.escrever_json

    def quebrar(caminho, documento):
        if caminho.name == arquivo and documento.get("status") != "failed":
            raise OSError("conteudo-privado")
        original(caminho, documento)

    monkeypatch.setattr(servico, "escrever_json", quebrar)
    output = tmp_path / "run"
    assert (
        servico.processar(tmp_path / "config.json", montar_config(), manifestos, "r", output) == 1
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    eventos = (output / "events.jsonl").read_text()
    assert "completed" not in eventos and "drift_alert" not in eventos
    assert "conteudo-privado" not in eventos


def test_pipeline_sem_probabilidades_nao_inventa_degradacao(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    class SemProbabilidade:
        def predict(self, base):
            return np.zeros(len(base), dtype=int)

    manifestos = preparar_manifestos(tmp_path)
    monkeypatch.setattr(servico, "carregar_pipeline", lambda _: SemProbabilidade())
    output = tmp_path / "run"
    codigo = servico.processar(tmp_path / "config.json", montar_config(), manifestos, "r", output)
    assert codigo == 0
    comparacao = json.loads((output / "performance.json").read_text())["comparison"]
    assert comparacao["degradation_detected"] is None
    assert comparacao["availability"] == "not_available"
