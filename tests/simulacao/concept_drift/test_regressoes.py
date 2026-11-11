"""Regressoes de integridade, concorrencia e falhas do produtor de Concept Drift."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from monitoramento.infraestrutura.arquivos import sha256_arquivo
from monitoramento.simulacao.concept_drift import servico

from .apoio import escrever_config, gerar_base, montar_config


def argumentos(tmp_path: Path) -> dict:
    origem = tmp_path / "source.csv"
    gerar_base().to_csv(origem, index=False)
    config = montar_config()
    config["base"]["path"] = str(origem)
    return dict(
        config_path=escrever_config(tmp_path, config),
        source_path=None,
        analysis_dir=tmp_path / "analysis",
        batch_id="batch",
        run_id="run",
        output_dir=tmp_path / "run",
        seed=42,
    )


def snapshot(pasta: Path) -> dict:
    return {str(p.relative_to(pasta)): p.read_bytes() for p in pasta.rglob("*") if p.is_file()}


@pytest.mark.parametrize("colisao", ["lote", "execucao"])
def test_colisao_preserva_publicacao(tmp_path: Path, colisao: str) -> None:
    args = argumentos(tmp_path)
    assert servico.gerar(**args) == 0
    antes = snapshot(tmp_path)
    if colisao == "lote":
        args.update(run_id="outro", output_dir=tmp_path / "outro")
    else:
        args["batch_id"] = "outro"
    args["seed"] = 43
    assert servico.gerar(**args) == 1
    assert snapshot(tmp_path) == antes


@pytest.mark.parametrize("mesma_execucao", [True, False])
def test_concorrencia_tem_um_unico_produtor(tmp_path: Path, mesma_execucao: bool) -> None:
    args = argumentos(tmp_path)
    barreira = Barrier(2)

    def executar(indice: int) -> int:
        chamada = {**args, "seed": 42 + indice}
        if not mesma_execucao:
            chamada.update(run_id=f"run-{indice}", output_dir=tmp_path / f"run-{indice}")
        barreira.wait(timeout=10)
        return servico.gerar(**chamada)

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(executar, [0, 1])) == [0, 1]
    manifestos = [json.loads(p.read_text()) for p in tmp_path.rglob("manifest.json")]
    sucesso = [m for m in manifestos if m["status"] == "success"]
    assert len(sucesso) == 1
    for item in sucesso[0]["outputs"]:
        assert sha256_arquivo(Path(item["path"])) == item["sha256"]


@pytest.mark.parametrize("falha", ["origem", "csv", "manifesto"])
def test_falha_publica_diagnostico_sem_saidas(tmp_path: Path, monkeypatch, falha: str) -> None:
    args = argumentos(tmp_path)
    if falha == "origem":
        args["source_path"] = tmp_path / "ausente.csv"
    elif falha == "csv":

        def quebrar(*_args):
            raise OSError("conteudo-privado")

        monkeypatch.setattr(servico, "escrever_artefato", quebrar)
    else:
        original = servico.escrever_json

        def quebrar(caminho, documento):
            if caminho.name == "manifest.json" and documento["status"] == "success":
                raise OSError("conteudo-privado")
            original(caminho, documento)

        monkeypatch.setattr(servico, "escrever_json", quebrar)
    assert servico.gerar(**args) == 1
    manifesto = json.loads((args["output_dir"] / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["outputs"] == []
    eventos = (args["output_dir"] / "events.jsonl").read_text()
    assert [json.loads(linha)["event"] for linha in eventos.splitlines()] == ["started", "failed"]
    assert "conteudo-privado" not in eventos
    assert json.loads((args["output_dir"] / "metrics.json").read_text())["measurements"] == []


@pytest.mark.parametrize("rotulo", ["0.9", "1.9", "-1", "2", "inf", "NaN", ""])
def test_simulacao_rejeita_rotulos_invalidos(tmp_path: Path, rotulo: str) -> None:
    args = argumentos(tmp_path)
    base = gerar_base()
    base.loc[0, "loan_status"] = rotulo
    base.to_csv(tmp_path / "source.csv", index=False)
    assert servico.gerar(**args) == 1
    assert json.loads((args["output_dir"] / "manifest.json").read_text())["outputs"] == []


def test_configuracao_invalida_nao_inicia_execucao(tmp_path: Path) -> None:
    args = argumentos(tmp_path)
    config = montar_config()
    config["target"]["positive_value"] = 0
    escrever_config(tmp_path, config)
    assert servico.gerar(**args) == 1
    assert not args["output_dir"].exists()
