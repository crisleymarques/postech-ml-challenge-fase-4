"""Testes da simulacao de Data Drift separados por responsabilidade."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento import validacao
from monitoramento.infraestrutura import arquivos
from monitoramento.simulacao.data_drift import artefatos, servico
from monitoramento.simulacao.data_drift import config as config_mod

from .apoio import (
    configurar,
    escrever_config,
    escrever_origem,
    executar,
    gerar_origem,
    hashes_diretorio,
    montar_config,
    parametros_geracao,
)


def test_carregar_base_rejeita_origem_ausente(tmp_path: Path) -> None:
    with pytest.raises(config_mod.ErroSimulacao, match="nao encontrada"):
        artefatos.carregar_base(tmp_path / "ausente.csv")


def test_gerar_publica_artefatos_e_manifesto(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)
    output_dir = tmp_path / "runs" / "run-sim"

    codigo = servico.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="batch-1",
        run_id="run-sim",
        output_dir=output_dir,
        seed=None,
    )

    assert codigo == config_mod.CODIGO_SUCESSO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["stage"] == "simulacao"
    assert manifesto["status"] == "success"
    assert manifesto["scenario_id"] == "data-drift"
    assert manifesto["batch_id"] == "batch-1"
    assert manifesto["seed"] == 42
    assert manifesto["periodo"]["shift_months"] == 24
    assert manifesto["periodo"]["origem"]["min"] == "Jul-2017"
    assert manifesto["periodo"]["simulado"]["min"] == "Jul-2019"
    assert manifesto["variaveis_alteradas"] == ["annual_inc", "dti", "revol_util"]
    assert len(manifesto["transformacoes"]) == 4
    assert "jitter" in manifesto["parametros"]

    saidas = {item["artifact_type"]: item for item in manifesto["outputs"]}
    assert set(saidas) == {"features", "labels"}
    for item in saidas.values():
        assert Path(item["path"]).is_file()
        assert arquivos.sha256_arquivo(Path(item["path"])) == item["sha256"]

    features = pd.read_csv(saidas["features"]["path"], dtype=str)
    rotulos = pd.read_csv(saidas["labels"]["path"], dtype=str)
    assert "loan_status" not in features.columns
    assert list(rotulos.columns) == ["record_id", "loan_status"]
    assert features["record_id"].is_unique

    metricas = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    nomes = {item["name"] for item in metricas["measurements"]}
    assert nomes == {"linhas_geradas", "variaveis_alteradas", "deslocamento_meses"}
    assert (output_dir / "events.jsonl").is_file()
    assert (output_dir / "config.json").is_file()


def test_gerar_e_reproduzivel(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)

    codigos = []
    hashes = []
    for execucao in ("a", "b"):
        run_dir = tmp_path / "runs" / f"run-{execucao}"
        codigos.append(
            servico.gerar(
                config_path=config_path,
                source_path=origem,
                analysis_dir=tmp_path / "analysis" / execucao,
                batch_id="batch-1",
                run_id=f"run-{execucao}",
                output_dir=run_dir,
                seed=None,
            )
        )
        hashes.append(
            arquivos.sha256_arquivo(
                tmp_path / "analysis" / execucao / "data-drift" / "batch-1" / "features.csv"
            )
        )

    assert codigos == [config_mod.CODIGO_SUCESSO, config_mod.CODIGO_SUCESSO]
    assert hashes[0] == hashes[1]


def test_base_de_origem_preservada(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)
    antes = arquivos.sha256_arquivo(origem)

    servico.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="batch-1",
        run_id="run-sim",
        output_dir=tmp_path / "runs" / "run-sim",
        seed=None,
    )

    assert arquivos.sha256_arquivo(origem) == antes


def test_dataset_simulado_difere_da_referencia(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)

    servico.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="batch-1",
        run_id="run-sim",
        output_dir=tmp_path / "runs" / "run-sim",
        seed=None,
    )

    base = gerar_origem()
    simulado = pd.read_csv(tmp_path / "analysis" / "data-drift" / "batch-1" / "features.csv")
    alteradas = [
        coluna
        for coluna in ("annual_inc", "dti", "revol_util")
        if not simulado[coluna].astype(float).equals(base[coluna].astype(float))
    ]
    assert len(alteradas) >= 2
    assert simulado["issue_d"].eq("Jul-2019").all()
    assert set(simulado["record_id"]) == set(base["record_id"])


@pytest.mark.parametrize("colisao", ["execucao", "lote"])
def test_gerar_preserva_artefatos_publicados_na_colisao(tmp_path: Path, colisao: str) -> None:
    parametros = parametros_geracao(tmp_path)
    assert servico.gerar(**parametros) == config_mod.CODIGO_SUCESSO
    anteriores = hashes_diretorio(tmp_path)
    if colisao == "lote":
        parametros.update(run_id="run-2", output_dir=tmp_path / "runs" / "run-2")
    parametros["seed"] = 7

    assert servico.gerar(**parametros) == config_mod.CODIGO_ERRO
    assert hashes_diretorio(tmp_path) == anteriores


@pytest.mark.parametrize("colisao", ["execucao", "lote"])
def test_reserva_exclusiva_rejeita_colisao_apos_prechecagem(
    tmp_path: Path, monkeypatch, colisao: str
) -> None:
    parametros = parametros_geracao(tmp_path)
    destino = (
        parametros["output_dir"]
        if colisao == "execucao"
        else tmp_path / "analysis" / "data-drift" / "batch-1"
    )
    mkdir = Path.mkdir
    concorrente = destino / "arquivo-do-concorrente"

    def reservar(caminho, *args, **kwargs):
        if caminho == destino:
            mkdir(caminho, parents=True, exist_ok=False)
            concorrente.write_text("preservar", encoding="utf-8")
        return mkdir(caminho, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", reservar)

    assert servico.gerar(**parametros) == config_mod.CODIGO_ERRO
    assert concorrente.read_text(encoding="utf-8") == "preservar"
    assert list(destino.iterdir()) == [concorrente]


def test_config_invalida_nao_reserva_diretorios(tmp_path: Path) -> None:
    parametros = parametros_geracao(tmp_path)
    config = montar_config()
    config.pop("scenario_id")
    escrever_config(tmp_path, config)

    assert servico.gerar(**parametros) == config_mod.CODIGO_ERRO
    assert not parametros["output_dir"].exists()
    assert not parametros["analysis_dir"].exists()


@pytest.mark.parametrize("falha", ["origem_ausente", "csv_malformado"])
def test_falha_de_leitura_publica_resultado_terminal(tmp_path: Path, capsys, falha: str) -> None:
    parametros = parametros_geracao(tmp_path)
    origem = parametros["source_path"]
    if falha == "origem_ausente":
        origem.unlink()
    else:
        origem.write_text('record_id,issue_d\n"dado-sigiloso-sem-fechamento', encoding="utf-8")
    parametros["seed"] = 7

    assert servico.gerar(**parametros) == config_mod.CODIGO_ERRO

    output_dir = parametros["output_dir"]
    manifesto = json.loads((output_dir / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["error_code"] == "ERRO_ENTRADA"
    assert manifesto["inputs"] == manifesto["outputs"] == []
    assert manifesto["seed"] == 7
    assert manifesto["finished_at"] >= manifesto["started_at"]
    config = json.loads(Path(manifesto["config_path"]).read_text())
    assert config["drift"]["seed"] == 7
    assert config["base"]["path"] == origem.as_posix()
    assert config["outputs"]["analysis_dir"] == parametros["analysis_dir"].as_posix()
    eventos = [
        json.loads(linha) for linha in (output_dir / "events.jsonl").read_text().splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "failed"]
    assert eventos[-1]["level"] == "ERROR"
    assert eventos[-1]["error_code"] == manifesto["error_code"]
    assert json.loads((output_dir / "metrics.json").read_text())["measurements"] == []
    saida = capsys.readouterr()
    assert saida.out == ""
    assert "ERRO_ENTRADA" in saida.err
    assert "dado-sigiloso" not in saida.err


@pytest.mark.parametrize("arquivo", ["labels.csv", "metrics.json", "manifest.json"])
def test_falha_de_publicacao_nao_aprova_artefatos(
    tmp_path: Path, monkeypatch, arquivo: str
) -> None:
    parametros = parametros_geracao(tmp_path)
    gravacoes = []
    escrever_json = servico.escrever_json
    escrever_csv = servico.escrever_artefato
    falhou = False

    def gravar_json(caminho, conteudo):
        nonlocal falhou
        if caminho.name == arquivo and not falhou:
            falhou = True
            # Simula inclusive um manifesto de sucesso escrito antes do erro.
            escrever_json(caminho, conteudo)
            raise OSError("Falha simulada com dado-sigiloso")
        escrever_json(caminho, conteudo)
        gravacoes.append(caminho.name)

    def gravar_csv(caminho, base):
        nonlocal falhou
        if caminho.name == arquivo and not falhou:
            falhou = True
            raise OSError("Falha simulada com dado-sigiloso")
        escrever_csv(caminho, base)

    monkeypatch.setattr(servico, "escrever_json", gravar_json)
    monkeypatch.setattr(servico, "escrever_artefato", gravar_csv)

    assert servico.gerar(**parametros) == config_mod.CODIGO_ERRO
    assert falhou
    output_dir = parametros["output_dir"]
    manifesto = json.loads((output_dir / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["error_code"] == "ERRO_TECNICO"
    assert manifesto["outputs"] == []
    assert json.loads((output_dir / "metrics.json").read_text())["measurements"] == []
    assert [
        json.loads(linha)["event"]
        for linha in (output_dir / "events.jsonl").read_text().splitlines()
    ] == ["started", "failed"]
    assert gravacoes[-1] == "manifest.json"


def test_falha_persistente_reporta_erro_controlado(tmp_path: Path, monkeypatch, capsys) -> None:
    parametros = parametros_geracao(tmp_path)

    def falhar(caminho, conteudo):
        raise OSError("dado-sigiloso")

    monkeypatch.setattr(servico, "escrever_json", falhar)

    assert servico.gerar(**parametros) == config_mod.CODIGO_ERRO
    assert not (parametros["output_dir"] / "manifest.json").exists()
    saida = capsys.readouterr()
    assert saida.out == ""
    assert "Falha ao persistir" in saida.err
    assert "dado-sigiloso" not in saida.err


def test_publicacao_de_sucesso_fecha_com_manifesto(tmp_path: Path, monkeypatch) -> None:
    parametros = parametros_geracao(tmp_path)
    parametros["seed"] = 7
    gravacoes = []
    escrever_json = servico.escrever_json
    escrever_eventos = servico.escrever_eventos

    def gravar_json(caminho, conteudo):
        escrever_json(caminho, conteudo)
        gravacoes.append(caminho.name)

    def gravar_eventos(caminho, eventos):
        escrever_eventos(caminho, eventos)
        gravacoes.append(caminho.name)

    monkeypatch.setattr(servico, "escrever_json", gravar_json)
    monkeypatch.setattr(servico, "escrever_eventos", gravar_eventos)

    assert servico.gerar(**parametros) == config_mod.CODIGO_SUCESSO
    assert gravacoes[:2] == ["config.json", "events.jsonl"]
    assert gravacoes[-1] == "manifest.json"
    config = json.loads((parametros["output_dir"] / "config.json").read_text())
    assert config["drift"]["seed"] == 7


@pytest.mark.parametrize("configuracao", ["shift_ausente", "jitter_invalido", "tempo_invalido"])
def test_config_invalida_para_manifesto_falha_antes_da_reserva(
    tmp_path: Path, configuracao: str
) -> None:
    parametros = parametros_geracao(tmp_path)
    config = montar_config()
    if configuracao == "shift_ausente":
        config["time"].pop("shift_months")
    elif configuracao == "jitter_invalido":
        config["drift"]["jitter"] = "invalido"
    else:
        config["time"] = "invalido"
    escrever_config(tmp_path, config)

    assert servico.gerar(**parametros) == config_mod.CODIGO_ERRO
    assert not parametros["output_dir"].exists()
    assert not parametros["analysis_dir"].exists()


@pytest.mark.parametrize("coluna", ["dti", "revol_util"])
@pytest.mark.parametrize("invalido", ["dado-sigiloso", "NaN", "inf", "-inf", " "])
def test_numero_invalido_nao_vira_ausencia_aprovada(
    tmp_path: Path, capsys, base, config, coluna: str, invalido: str
) -> None:
    base.loc[0, coluna] = invalido
    origem = tmp_path / "origem.csv"
    base.to_csv(origem, index=False)
    hash_origem = arquivos.sha256_arquivo(origem)
    # Estas features permitem ausencia, mas devem rejeitar conteudo invalido.
    contrato = validacao.construir_contrato(
        {
            "profile": "production",
            "rules": {
                "identifier": {"column": "record_id", "pattern": "^LC[0-9]{7}$"},
                "target": {"column": "loan_status", "domain": [0, 1]},
                "not_null": ["record_id"],
                "unique": ["record_id"],
                "ranges": {"dti": [0, 500], "revol_util": [0, 1000]},
                "categories": {},
            },
        }
    )
    _, violacoes = validacao.validar_artefato(origem, contrato)
    # O leitor da validacao considera "NaN" ausencia; a simulacao le o texto
    # literal com na_filter=False e tambem deve recusar esse numero nao finito.
    if invalido != "NaN":
        assert violacoes
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "run"

    codigo = servico.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="invalid-numeric",
        run_id="invalid-numeric",
        output_dir=output_dir,
        seed=None,
    )

    assert codigo == config_mod.CODIGO_ERRO
    manifesto = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifesto["status"] == "failed"
    assert manifesto["error_code"] == "ERRO_ENTRADA"
    assert manifesto["outputs"] == []
    assert not list((tmp_path / "analysis").rglob("*.csv"))
    eventos = [
        json.loads(linha)
        for linha in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [evento["event"] for evento in eventos] == ["started", "failed"]
    assert eventos[-1]["level"] == "ERROR"
    assert eventos[-1]["error_code"] == "ERRO_ENTRADA"
    assert json.loads((output_dir / "metrics.json").read_text())["measurements"] == []
    assert arquivos.sha256_arquivo(origem) == hash_origem
    saida = capsys.readouterr()
    assert saida.out == ""
    assert "ERRO_ENTRADA" in saida.err
    assert "dado-sigiloso" not in saida.err
    assert "dado-sigiloso" not in (output_dir / "events.jsonl").read_text()


@pytest.mark.parametrize("cenario", ["neutro", "uma_feature", "zeros", "limite", "ausentes"])
def test_rejeita_menos_de_duas_distribuicoes_alteradas(tmp_path: Path, cenario: str) -> None:
    config, parametros = configurar(tmp_path)
    variaveis = config["drift"]["variables"]
    if cenario == "neutro":
        variaveis["annual_inc"]["factor"] = 1
        variaveis["dti"]["delta"] = 0
    elif cenario == "uma_feature":
        variaveis["dti"]["delta"] = 0
    else:
        origem = Path(config["base"]["path"])
        base = pd.read_csv(origem)
        if cenario == "zeros":
            base["annual_inc"] = 0
        elif cenario == "limite":
            base["dti"] = 500
        else:
            base["annual_inc"] = ""
        base.to_csv(origem, index=False)
    antes = arquivos.sha256_arquivo(Path(config["base"]["path"]))

    assert executar(config, parametros) == config_mod.CODIGO_ERRO

    run_dir = parametros["output_dir"]
    manifesto = json.loads((run_dir / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["outputs"] == []
    assert manifesto["variaveis_alteradas"] == []
    assert manifesto["error_code"] == "ERRO_ENTRADA"
    assert json.loads((run_dir / "metrics.json").read_text())["measurements"] == []
    eventos = [json.loads(linha) for linha in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [evento["event"] for evento in eventos] == ["started", "failed"]
    assert not list(Path(config["outputs"]["analysis_dir"]).rglob("*.csv"))
    assert arquivos.sha256_arquivo(Path(config["base"]["path"])) == antes


def test_reordenar_valores_nao_altera_distribuicao(tmp_path: Path) -> None:
    config, parametros = configurar(tmp_path)
    origem = Path(config["base"]["path"])
    base = pd.read_csv(origem)
    base["dti"] = [-1, 1]
    base.to_csv(origem, index=False)
    config["drift"]["variables"]["dti"] = {"operation": "scale", "factor": -1, "clip": [-100, 100]}

    assert executar(config, parametros) == config_mod.CODIGO_ERRO
    manifesto = json.loads((parametros["output_dir"] / "manifest.json").read_text())
    assert manifesto["status"] == "failed"
    assert manifesto["outputs"] == []


@pytest.mark.parametrize("terceira_ausente", [False, True])
def test_sucesso_informa_apenas_distribuicoes_alteradas(
    tmp_path: Path, terceira_ausente: bool
) -> None:
    config, parametros = configurar(tmp_path)
    if terceira_ausente:
        origem = Path(config["base"]["path"])
        base = pd.read_csv(origem)
        base["revol_util"] = ""
        base.to_csv(origem, index=False)

    assert executar(config, parametros) == config_mod.CODIGO_SUCESSO

    run_dir = parametros["output_dir"]
    manifesto = json.loads((run_dir / "manifest.json").read_text())
    assert manifesto["status"] == "success"
    assert manifesto["variaveis_alteradas"] == ["annual_inc", "dti"]
    assert set(manifesto["parametros"]["variables"]) == {"annual_inc", "dti"}
    assert len(manifesto["transformacoes"]) == 3
    metricas = json.loads((run_dir / "metrics.json").read_text())["measurements"]
    assert next(m["value"] for m in metricas if m["name"] == "variaveis_alteradas") == 2
    features = next(item for item in manifesto["outputs"] if item["artifact_type"] == "features")
    base_simulada = pd.read_csv(features["path"], dtype=str, na_filter=False)
    assert list(base_simulada["issue_d"]) == ["Jul-2019", "Dec-2019"]
