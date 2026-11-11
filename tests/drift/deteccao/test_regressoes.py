"""Regressoes de integridade, cobertura e publicacao da deteccao."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from monitoramento.drift.deteccao import adaptador_evidently as adaptador
from monitoramento.drift.deteccao import regras, servico
from monitoramento.infraestrutura import arquivos, manifestos

from .apoio import FEATURES, SnapshotFalso, metrica_drift, montar_config, preparar_manifestos
from .test_cli import executar_cli


def ler(caminho: Path) -> dict:
    return json.loads(caminho.read_text())


def atualizar_csv(manifesto: dict, transformar) -> None:
    item = manifesto["outputs"][0]
    caminho = Path(item["path"])
    transformar(pd.read_csv(caminho)).to_csv(caminho, index=False)
    item["sha256"] = arquivos.sha256_arquivo(caminho)


def executar(tmp_path, manifestos, config=None, nome="run"):
    return servico.processar(
        tmp_path / "config.json", config or montar_config(), manifestos, nome, tmp_path / nome
    )


class Tabela(HTMLParser):
    def __init__(self):
        super().__init__()
        self.linhas = []
        self.linha = None
        self.celula = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.linha = []
        if tag == "td":
            self.celula = ""

    def handle_data(self, data):
        if self.celula is not None:
            self.celula += data

    def handle_endtag(self, tag):
        if tag == "td":
            self.linha.append(self.celula)
            self.celula = None
        if tag == "tr" and self.linha:
            self.linhas.append(self.linha)


@pytest.mark.parametrize(
    "metodo,valor,limiar,esperado",
    [
        ("K-S p_value", 0.1, 0.1, False),
        ("K-S p_value", 0.09, 0.1, True),
        ("K-S p_value", 0.11, 0.1, False),
        ("Wasserstein distance", 0.9, 0.9, False),
        ("Wasserstein distance", 0.89, 0.9, False),
        ("Wasserstein distance", 0.91, 0.9, True),
    ],
)
def test_html_json_metricas_alertas_usam_a_mesma_decisao(
    tmp_path, monkeypatch, metodo, valor, limiar, esperado
):
    entradas = preparar_manifestos(tmp_path)
    config = montar_config()
    config["method"]["p_value_threshold"] = limiar
    config["method"]["distance_threshold"] = limiar
    snapshot = SnapshotFalso([metrica_drift(f, valor, metodo) for f in FEATURES])
    monkeypatch.setattr(adaptador, "executar_evidently", lambda *_: snapshot)
    assert executar(tmp_path, entradas, config) == 0
    doc = ler(tmp_path / "run/drift.json")
    html = (tmp_path / "run/drift.html").read_text()
    tabela = Tabela()
    tabela.feed(html)
    for linha, item in zip(tabela.linhas, doc["features"], strict=True):
        assert linha[:4] == [item["feature"], metodo, str(valor), str(limiar)]
        assert linha[4] == ("Drift" if esperado else "Sem drift")
        assert item["drift_detected"] is esperado
    assert doc["summary"]["dataset_drift"] is esperado
    assert ler(tmp_path / "run/manifest.json")["drift"] == doc["summary"]
    medidas = ler(tmp_path / "run/metrics.json")["measurements"]
    assert medidas[1]["value"] == (5 if esperado else 0)
    eventos = [
        json.loads(linha) for linha in (tmp_path / "run/events.jsonl").read_text().splitlines()
    ]
    assert any(e["event"] == "drift_alert" for e in eventos) is esperado


@pytest.mark.parametrize("lado", [0, 1])
def test_feature_ausente_falha_sem_aprovacao_parcial(tmp_path, lado):
    entradas = preparar_manifestos(tmp_path)
    atualizar_csv(entradas[lado], lambda df: df.drop(columns=["annual_inc"]))
    assert executar(tmp_path, entradas) == 1
    assert ler(tmp_path / "run/manifest.json")["status"] == "failed"
    assert not (tmp_path / "run/drift.html").exists()


@pytest.mark.parametrize("lado,colunas", [(0, ["dti"]), (1, ["dti"]), (1, ["dti", "revol_util"])])
def test_colunas_nulas_preservam_cobertura_sem_bloquear_demais(tmp_path, lado, colunas):
    entradas = preparar_manifestos(tmp_path)
    atualizar_csv(entradas[lado], lambda df: df.assign(**dict.fromkeys(colunas, float("nan"))))
    assert executar(tmp_path, entradas) == 0
    doc = ler(tmp_path / "run/drift.json")
    assert len(doc["features"]) == 5
    assert doc["summary"]["evaluated_columns"] == 5 - len(colunas)
    assert doc["changed_variables"]["not_evaluated"] == colunas
    assert not set(colunas) & set(doc["changed_variables"]["not_detected"])
    for item in doc["features"]:
        if item["feature"] in colunas:
            assert item["drift_detected"] is None
            assert "sem valores" in item["reason"]
    assert "Nao avaliavel" in (tmp_path / "run/drift.html").read_text()


def test_nenhuma_feature_avaliavel_e_falha_controlada(tmp_path):
    entradas = preparar_manifestos(tmp_path)
    atualizar_csv(entradas[1], lambda df: df.assign(**dict.fromkeys(FEATURES, float("nan"))))
    assert executar(tmp_path, entradas) == 1
    assert ler(tmp_path / "run/manifest.json")["outputs"] == []
    assert not (tmp_path / "run/drift.json").exists()


@pytest.mark.parametrize("lado,indice", [(0, 0), (0, 1), (1, 0)])
@pytest.mark.parametrize("hash_invalido", [None, "", "xyz", "0" * 64])
def test_hash_obrigatorio_em_cada_entrada(tmp_path, lado, indice, hash_invalido):
    entradas = preparar_manifestos(tmp_path)
    item = entradas[lado]["outputs"][indice]
    if hash_invalido is None:
        item.pop("sha256")
    else:
        item["sha256"] = hash_invalido
    assert executar(tmp_path, entradas) == 1
    assert ler(tmp_path / "run/manifest.json")["status"] == "failed"
    assert not (tmp_path / "run/drift.html").exists()


def test_helper_preserva_compatibilidade_para_outros_consumidores(tmp_path):
    caminho = tmp_path / "origem"
    caminho.write_text("origem")
    assert manifestos.verificar_artefato({"path": str(caminho)}, erro=ValueError) == caminho


def test_diretorio_existente_nao_e_alterado_por_chamada_direta(tmp_path):
    entradas = preparar_manifestos(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "drift.json").write_text("relatorio anterior")
    antes = {p.name: p.read_bytes() for p in output.iterdir()}
    assert executar(tmp_path, entradas) == 1
    assert {p.name: p.read_bytes() for p in output.iterdir()} == antes


def test_dois_processos_disputando_destino_tem_um_unico_vencedor(tmp_path):
    preparar_manifestos(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(montar_config()))

    def chamar(nome):
        return executar_cli(
            "--config",
            str(config),
            "--input-manifest",
            str(tmp_path / "man-treino.json"),
            "--input-manifest",
            str(tmp_path / "man-simulacao.json"),
            "--run-id",
            nome,
            "--output-dir",
            str(tmp_path / "run"),
        )

    with ThreadPoolExecutor(2) as pool:
        resultados = list(pool.map(chamar, ["primeiro", "segundo"]))
    assert sorted(r.returncode for r in resultados) == [0, 1]
    manifesto = ler(tmp_path / "run/manifest.json")
    doc = ler(tmp_path / "run/drift.json")
    assert manifesto["run_id"] == doc["run_id"]
    for item in manifesto["outputs"]:
        assert arquivos.sha256_arquivo(Path(item["path"])) == item["sha256"]


@pytest.mark.parametrize("ponto", ["relatorio", "config", "metricas", "eventos", "manifesto"])
def test_falhas_de_publicacao_nunca_deixam_manifesto_de_sucesso(tmp_path, monkeypatch, ponto):
    entradas = preparar_manifestos(tmp_path)
    if ponto == "relatorio":

        def falhar(*_):
            raise OSError("falha no HTML")

        monkeypatch.setattr(servico, "salvar_html", falhar)
    elif ponto in ("config", "metricas"):
        original = servico.escrever_json

        def escrever(caminho, conteudo):
            if caminho.name == {"config": "config.json", "metricas": "metrics.json"}[ponto]:
                raise OSError("falha no JSON")
            return original(caminho, conteudo)

        monkeypatch.setattr(servico, "escrever_json", escrever)
    elif ponto == "eventos":
        original = servico.escrever_eventos

        def escrever(caminho, eventos):
            if any(e["event"] == "completed" for e in eventos):
                raise OSError("falha nos eventos finais")
            return original(caminho, eventos)

        monkeypatch.setattr(servico, "escrever_eventos", escrever)
    else:
        original = arquivos.os.replace

        def substituir(origem, destino):
            if ler(Path(origem)).get("status") == "success":
                raise OSError("falha na publicacao atomica")
            return original(origem, destino)

        monkeypatch.setattr(arquivos.os, "replace", substituir)
    assert executar(tmp_path, entradas) == 1
    manifesto = ler(tmp_path / "run/manifest.json")
    assert manifesto["status"] == "failed"
    assert manifesto["outputs"] == []
    eventos = [
        json.loads(linha) for linha in (tmp_path / "run/events.jsonl").read_text().splitlines()
    ]
    assert eventos[-1]["event"] == "failed"
    assert not any(e["event"] in ("completed", "drift_alert") for e in eventos)
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_manifesto_e_publicado_atomicamente_depois_dos_eventos(tmp_path, monkeypatch):
    entradas = preparar_manifestos(tmp_path)
    original = arquivos.os.replace

    def publicar(origem, destino):
        assert not Path(destino).exists()
        eventos = [
            json.loads(linha) for linha in (tmp_path / "run/events.jsonl").read_text().splitlines()
        ]
        assert [e["event"] for e in eventos] == ["started", "drift_alert", "completed"]
        assert ler(Path(origem))["status"] == "success"
        return original(origem, destino)

    monkeypatch.setattr(arquivos.os, "replace", publicar)
    assert executar(tmp_path, entradas) == 0


def test_reexecucoes_preservam_decisoes_e_entradas(tmp_path):
    entradas = preparar_manifestos(tmp_path)
    hashes = [i["sha256"] for m in entradas for i in m["outputs"]]
    assert executar(tmp_path, entradas, nome="um") == 0
    assert executar(tmp_path, entradas, nome="dois") == 0
    a, b = ler(tmp_path / "um/drift.json"), ler(tmp_path / "dois/drift.json")
    for campo in ["features", "summary", "changed_variables", "method", "samples"]:
        assert a[campo] == b[campo]
    assert [
        arquivos.sha256_arquivo(Path(i["path"])) for m in entradas for i in m["outputs"]
    ] == hashes


def test_proporcao_e_comparada_antes_do_arredondamento():
    assert (
        regras.resumir([{"feature": "x", "drift_detected": v} for v in [True, False]])[
            "dataset_drift"
        ]
        is False
    )

    # 0.50000025 arredonda para 0.5, mas ainda deve emitir alerta.
    class Resultados:
        def __iter__(self):
            for i in range(2_000_001):
                yield {"feature": "x", "drift_detected": i < 1_000_001}

    assert regras.resumir(Resultados())["dataset_drift"] is True
