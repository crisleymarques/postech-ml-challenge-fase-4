"""Publicacao terminal e preservacao dos artefatos da analise."""

import json
from pathlib import Path

import pytest

from monitoramento.drift.analise import servico
from monitoramento.observabilidade.logs import gerar

from .apoio import montar_config, preparar_manifestos


def conferir_falha(output: Path) -> dict:
    manifesto = json.loads((output / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["error_code"] == "ANALISE_INVALIDA"
    assert manifesto["outputs"] == []
    assert manifesto["finished_at"] >= manifesto["started_at"]
    assert json.loads((output / "metrics.json").read_text())["measurements"] == []
    assert (output / "config.json").is_file()
    eventos = [json.loads(linha) for linha in (output / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in eventos] == ["started", "failed"]
    return manifesto


@pytest.mark.parametrize("defeito", ["dataset", "hash", "ausente", "json", "rejeitado"])
def test_falhas_de_entrada_publicam_manifesto(tmp_path: Path, defeito: str) -> None:
    config = montar_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    manifestos = preparar_manifestos(tmp_path)
    if defeito == "dataset":
        config["dataset_id"] = "outro"
    elif defeito == "hash":
        manifestos[1]["outputs"][0]["sha256"] = "0" * 64
    else:
        path = tmp_path / "man-simulacao.json"
        if defeito == "ausente":
            path.unlink()
        elif defeito == "json":
            path.write_text("{")
        else:
            doc = json.loads(path.read_text())
            doc["status"] = "rejected"
            path.write_text(json.dumps(doc))
        manifestos = [tmp_path / "man-treino.json", path]
    output = tmp_path / "run"
    assert servico.processar(config_path, config, manifestos, "run", output) == 1
    conferir_falha(output)


@pytest.mark.parametrize(
    "arquivo", ["analise.json", "metrics.json", "events.jsonl", "manifest.json"]
)
def test_falha_de_publicacao_nao_deixa_sucesso(tmp_path: Path, monkeypatch, arquivo: str) -> None:
    config = montar_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    manifestos = preparar_manifestos(tmp_path)
    output = tmp_path / "run"
    real_json = servico.escrever_json
    real_eventos = servico.escrever_eventos
    real_atomico = getattr(servico, "escrever_json_atomico", None)
    falhou = False

    def interceptar(original, caminho, conteudo):
        nonlocal falhou
        terminal = caminho.name != "events.jsonl" or any(
            e["event"] == "completed" for e in conteudo
        )
        if caminho.name == arquivo and terminal and not falhou:
            falhou = True
            raise OSError("falha de escrita simulada")
        return original(caminho, conteudo)

    monkeypatch.setattr(servico, "escrever_json", lambda p, v: interceptar(real_json, p, v))
    monkeypatch.setattr(servico, "escrever_eventos", lambda p, v: interceptar(real_eventos, p, v))
    if real_atomico:
        monkeypatch.setattr(
            servico, "escrever_json_atomico", lambda p, v: interceptar(real_atomico, p, v)
        )
    assert servico.processar(path, config, manifestos, "run", output) == 1
    assert falhou
    manifesto = conferir_falha(output)
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["scenario_id"] == "data-drift"
    assert len(manifesto["inputs"]) == 3


def test_destino_existente_preserva_publicacao(tmp_path: Path) -> None:
    config = montar_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    manifestos = preparar_manifestos(tmp_path)
    output = tmp_path / "run"
    assert servico.processar(path, config, manifestos, "run", output) == 0
    anteriores = {p.name: p.read_bytes() for p in output.iterdir()}
    assert servico.processar(path, config, manifestos, "run", output) == 1
    assert {p.name: p.read_bytes() for p in output.iterdir()} == anteriores


def test_calculo_falha_preserva_identidades(tmp_path: Path, monkeypatch) -> None:
    config = montar_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    manifestos = preparar_manifestos(tmp_path)
    output = tmp_path / "run"

    def falhar(*args, **kwargs):
        raise RuntimeError("calculo indisponivel")

    monkeypatch.setattr(servico, "montar_resultados", falhar)
    assert servico.processar(path, config, manifestos, "run", output) == 1
    manifesto = conferir_falha(output)
    assert manifesto["model_id"] == "modelo-teste"
    assert manifesto["batch_id"] == "batch-1"
    assert len(manifesto["inputs"]) == 3


def test_manifesto_de_sucesso_publicado_apos_eventos(tmp_path: Path, monkeypatch) -> None:
    config = montar_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    manifestos = preparar_manifestos(tmp_path)
    output = tmp_path / "run"
    publicar = servico.escrever_json_atomico
    observado = False

    def verificar(caminho, manifesto):
        nonlocal observado
        assert manifesto["status"] == "success"
        assert not caminho.exists()
        assert all(
            (output / nome).is_file()
            for nome in ("config.json", "analise.json", "metrics.json", "events.jsonl")
        )
        eventos = [
            json.loads(linha) for linha in (output / "events.jsonl").read_text().splitlines()
        ]
        assert eventos[-1]["event"] == "completed"
        observado = True
        publicar(caminho, manifesto)

    monkeypatch.setattr(servico, "escrever_json_atomico", verificar)
    assert servico.processar(path, config, manifestos, "run", output) == 0
    assert observado


def test_config_invalida_nao_reserva_diretorio(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("method")
    output = tmp_path / "run"
    assert servico.processar(tmp_path / "config.json", config, [], "run", output) == 1
    assert not output.exists()


def test_falha_persistente_nos_eventos_permanece_nos_logs(tmp_path: Path, monkeypatch) -> None:
    def falhar(*args):
        raise OSError("events.jsonl indisponivel")

    monkeypatch.setattr(servico, "escrever_eventos", falhar)
    runs = tmp_path / "runs"
    output = runs / "run"
    assert servico.processar(tmp_path / "config.json", montar_config(), [], "run", output) == 1
    assert not (output / "events.jsonl").exists()
    manifesto = json.loads((output / "manifest.json").read_text())

    resumo = gerar(runs, tmp_path / "logs")

    assert resumo["execucoes"]["por_status"] == {"failed": 1}
    assert len(resumo["falhas"]) == 1
    assert resumo["falhas"][0]["run_id"] == manifesto["run_id"]
    assert resumo["falhas"][0]["stage"] == manifesto["stage"]
    assert resumo["falhas"][0]["error_code"] == "ANALISE_INVALIDA"


@pytest.mark.parametrize("defeito", ["rejeitado", "versao"])
def test_api_python_valida_manifestos(tmp_path: Path, defeito: str) -> None:
    config = montar_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    manifestos = preparar_manifestos(tmp_path)
    if defeito == "rejeitado":
        manifestos[1]["status"] = "rejected"
    else:
        manifestos[1]["contract_version"] = "99.0"
    output = tmp_path / "run"
    assert servico.processar(path, config, manifestos, "run", output) == 1
    conferir_falha(output)
