"""Testes da simulacao temporal de Data Drift (issue F2-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from monitoramento import simulacao_drift as sd


def montar_config() -> dict:
    """Contrato minimo compativel com as variaveis simuladas."""
    return {
        "contract_version": "1.0",
        "stage": "simulacao",
        "dataset_id": "teste",
        "schema_version": "1.0",
        "scenario_id": "data-drift",
        "base": {"artifact_type": "origem", "path": "origem.csv"},
        "time": {"column": "issue_d", "format": "%b-%Y", "shift_months": 24},
        "target": {
            "column": "loan_status",
            "positive_class": "Charged Off",
            "negative_class": "Fully Paid",
        },
        "drift": {
            "seed": 42,
            "jitter": 0.05,
            "variables": {
                "annual_inc": {"operation": "scale", "factor": 0.8, "clip": [0, 10000000]},
                "dti": {"operation": "shift", "delta": 6.0, "clip": [0, 500]},
                "revol_util": {"operation": "scale", "factor": 1.35, "clip": [0, 1000]},
            },
        },
        "outputs": {"analysis_dir": "data/analysis"},
    }


def gerar_origem(quantidade: int = 40) -> pd.DataFrame:
    """Cria uma base de producao simulada determinista."""
    linhas = []
    for indice in range(quantidade):
        linhas.append(
            {
                "record_id": f"LC{indice + 1:07d}",
                "loan_amnt": "10000.0",
                "term": "36 months",
                "annual_inc": str(100000 + (indice % 5) * 1000),
                "issue_d": "Jul-2017",
                "loan_status": str(indice % 2),
                "dti": str(10 + (indice % 4)),
                "fico_range_low": "700.0",
                "fico_range_high": "704.0",
                "revol_util": "950.0" if indice == 0 else str(50 + (indice % 6) * 100),
                "home_ownership": "RENT",
            }
        )
    return pd.DataFrame(linhas)


def escrever_config(tmp_path: Path, config: dict) -> Path:
    caminho = tmp_path / "simulacao.json"
    caminho.write_text(json.dumps(config), encoding="utf-8")
    return caminho


def escrever_origem(tmp_path: Path, quantidade: int = 40) -> Path:
    caminho = tmp_path / "origem.csv"
    gerar_origem(quantidade).to_csv(caminho, index=False, encoding="utf-8")
    return caminho


def test_carregar_config_rejeita_chaves_ausentes(tmp_path: Path) -> None:
    config = montar_config()
    config.pop("scenario_id")
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="Chaves ausentes"):
        sd.carregar_config(caminho)


def test_carregar_config_rejeita_menos_de_duas_variaveis(tmp_path: Path) -> None:
    config = montar_config()
    config["drift"]["variables"] = {
        "annual_inc": {"operation": "scale", "factor": 0.8, "clip": [0, 10000000]}
    }
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="ao menos 2 variaveis"):
        sd.carregar_config(caminho)


def test_carregar_config_rejeita_operacao_invalida(tmp_path: Path) -> None:
    config = montar_config()
    config["drift"]["variables"]["annual_inc"]["operation"] = "media"
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="Operacao invalida"):
        sd.carregar_config(caminho)


def test_carregar_config_rejeita_versao_incompativel(tmp_path: Path) -> None:
    config = montar_config()
    config["contract_version"] = "2.0"
    caminho = escrever_config(tmp_path, config)

    with pytest.raises(ValueError, match="incompativel"):
        sd.carregar_config(caminho)


def test_carregar_base_rejeita_origem_ausente(tmp_path: Path) -> None:
    with pytest.raises(sd.ErroSimulacao, match="nao encontrada"):
        sd.carregar_base(tmp_path / "ausente.csv")


def test_deslocar_periodo_avanca_meses() -> None:
    serie = pd.Series(["Jul-2017", "Dec-2017"])

    deslocada = sd.deslocar_periodo(serie, "%b-%Y", 24)

    assert list(deslocada) == ["Jul-2019", "Dec-2019"]


def test_aplicar_drift_altera_media_e_respeita_limites() -> None:
    base = gerar_origem()
    variaveis = montar_config()["drift"]["variables"]

    ajustado, descritas = sd.aplicar_drift(base, variaveis, seed=42, jitter=0.05)

    assert [item["column"] for item in descritas] == ["annual_inc", "dti", "revol_util"]
    assert ajustado["annual_inc"].mean() < base["annual_inc"].astype(float).mean() * 0.9
    assert ajustado["dti"].mean() > base["dti"].astype(float).mean()
    assert ajustado["revol_util"].max() == pytest.approx(1000.0)
    assert ajustado["record_id"].equals(base["record_id"])


def test_aplicar_drift_preserva_ausentes() -> None:
    base = gerar_origem()
    base.loc[3, "revol_util"] = ""

    ajustado, descritas = sd.aplicar_drift(base, montar_config()["drift"]["variables"], 42, 0.05)

    assert ajustado.at[3, "revol_util"] == ""
    descricao = next(item for item in descritas if item["column"] == "revol_util")
    assert descricao["missing_preserved"] == 1


def test_aplicar_drift_rejeita_coluna_ausente() -> None:
    base = pd.DataFrame({"record_id": ["LC0000001"]})

    with pytest.raises(sd.ErroSimulacao, match="Coluna da alteracao ausente"):
        sd.aplicar_drift(
            base, {"annual_inc": {"operation": "shift", "delta": 1.0, "clip": [0, 10]}}, 1, 0.0
        )


def test_separar_features_rotulos_remove_alvo() -> None:
    base = gerar_origem()

    features, rotulos = sd.separar_features_rotulos(base, "loan_status")

    assert "loan_status" not in features.columns
    assert "record_id" in features.columns
    assert list(rotulos.columns) == ["record_id", "loan_status"]
    assert len(features) == len(rotulos) == len(base)


def test_gerar_publica_artefatos_e_manifesto(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)
    output_dir = tmp_path / "runs" / "run-sim"

    codigo = sd.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="batch-1",
        run_id="run-sim",
        output_dir=output_dir,
        seed=None,
    )

    assert codigo == sd.CODIGO_SUCESSO
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
        assert sd.sha256_arquivo(Path(item["path"])) == item["sha256"]

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
            sd.gerar(
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
            sd.sha256_arquivo(
                tmp_path / "analysis" / execucao / "data-drift" / "batch-1" / "features.csv"
            )
        )

    assert codigos == [sd.CODIGO_SUCESSO, sd.CODIGO_SUCESSO]
    assert hashes[0] == hashes[1]


def test_base_de_origem_preservada(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)
    antes = sd.sha256_arquivo(origem)

    sd.gerar(
        config_path=config_path,
        source_path=origem,
        analysis_dir=tmp_path / "analysis",
        batch_id="batch-1",
        run_id="run-sim",
        output_dir=tmp_path / "runs" / "run-sim",
        seed=None,
    )

    assert sd.sha256_arquivo(origem) == antes


def test_dataset_simulado_difere_da_referencia(tmp_path: Path) -> None:
    config_path = escrever_config(tmp_path, montar_config())
    origem = escrever_origem(tmp_path)

    sd.gerar(
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


def parametros_geracao(tmp_path: Path) -> dict:
    return {
        "config_path": escrever_config(tmp_path, montar_config()),
        "source_path": escrever_origem(tmp_path),
        "analysis_dir": tmp_path / "analysis",
        "batch_id": "batch-1",
        "run_id": "run-1",
        "output_dir": tmp_path / "runs" / "run-1",
        "seed": 42,
    }


def hashes_diretorio(diretorio: Path) -> dict[str, str]:
    return {
        caminho.relative_to(diretorio).as_posix(): sd.sha256_arquivo(caminho)
        for caminho in diretorio.rglob("*")
        if caminho.is_file()
    }


@pytest.mark.parametrize("colisao", ["execucao", "lote"])
def test_gerar_preserva_artefatos_publicados_na_colisao(tmp_path: Path, colisao: str) -> None:
    parametros = parametros_geracao(tmp_path)
    assert sd.gerar(**parametros) == sd.CODIGO_SUCESSO
    anteriores = hashes_diretorio(tmp_path)
    if colisao == "lote":
        parametros.update(run_id="run-2", output_dir=tmp_path / "runs" / "run-2")
    parametros["seed"] = 7

    assert sd.gerar(**parametros) == sd.CODIGO_ERRO
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

    assert sd.gerar(**parametros) == sd.CODIGO_ERRO
    assert concorrente.read_text(encoding="utf-8") == "preservar"
    assert list(destino.iterdir()) == [concorrente]


def test_config_invalida_nao_reserva_diretorios(tmp_path: Path) -> None:
    parametros = parametros_geracao(tmp_path)
    config = montar_config()
    config.pop("scenario_id")
    escrever_config(tmp_path, config)

    assert sd.gerar(**parametros) == sd.CODIGO_ERRO
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

    assert sd.gerar(**parametros) == sd.CODIGO_ERRO

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
    escrever_json = sd.escrever_json
    escrever_csv = sd.escrever_artefato
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

    monkeypatch.setattr(sd, "escrever_json", gravar_json)
    monkeypatch.setattr(sd, "escrever_artefato", gravar_csv)

    assert sd.gerar(**parametros) == sd.CODIGO_ERRO
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

    monkeypatch.setattr(sd, "escrever_json", falhar)

    assert sd.gerar(**parametros) == sd.CODIGO_ERRO
    assert not (parametros["output_dir"] / "manifest.json").exists()
    saida = capsys.readouterr()
    assert saida.out == ""
    assert "Falha ao persistir" in saida.err
    assert "dado-sigiloso" not in saida.err


def test_publicacao_de_sucesso_fecha_com_manifesto(tmp_path: Path, monkeypatch) -> None:
    parametros = parametros_geracao(tmp_path)
    parametros["seed"] = 7
    gravacoes = []
    escrever_json = sd.escrever_json
    escrever_eventos = sd.escrever_eventos

    def gravar_json(caminho, conteudo):
        escrever_json(caminho, conteudo)
        gravacoes.append(caminho.name)

    def gravar_eventos(caminho, eventos):
        escrever_eventos(caminho, eventos)
        gravacoes.append(caminho.name)

    monkeypatch.setattr(sd, "escrever_json", gravar_json)
    monkeypatch.setattr(sd, "escrever_eventos", gravar_eventos)

    assert sd.gerar(**parametros) == sd.CODIGO_SUCESSO
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

    assert sd.gerar(**parametros) == sd.CODIGO_ERRO
    assert not parametros["output_dir"].exists()
    assert not parametros["analysis_dir"].exists()
