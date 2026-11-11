"""Regressoes dos limites documentados e da publicacao das metricas de saude."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoramento.infraestrutura import arquivos
from monitoramento.observabilidade.metricas_saude import catalogo, classificacao, resultados

CATALOGO = Path(__file__).resolve().parents[3] / "configs/metricas_saude.json"


@pytest.mark.parametrize(
    ("metrica_id", "limiar", "passo", "abaixo", "igual", "acima"),
    [
        ("pipeline.volume_linhas", 1, 1, "atencao", "ok", "ok"),
        ("pipeline.taxa_sucesso", 0.9, 0.01, "atencao", "ok", "ok"),
        ("pipeline.taxa_sucesso", 0.5, 0.01, "critico", "atencao", "atencao"),
        ("modelo.roc_auc", 0.6, 0.01, "atencao", "ok", "ok"),
        ("modelo.roc_auc", 0.5, 0.01, "critico", "atencao", "atencao"),
        ("infra.espaco_livre_bytes", 1073741824, 1, "atencao", "ok", "ok"),
        ("infra.espaco_livre_bytes", 268435456, 1, "critico", "atencao", "atencao"),
        ("pipeline.duracao_s", 60, 0.01, "ok", "atencao", "atencao"),
        ("pipeline.duracao_s", 300, 0.01, "atencao", "critico", "critico"),
        ("drift.proporcao_drift", 0.1, 0.01, "ok", "atencao", "atencao"),
        ("drift.proporcao_drift", 0.5, 0.01, "atencao", "critico", "critico"),
    ],
)
@pytest.mark.parametrize("posicao", [-1, 0, 1])
def test_limites_do_catalogo_real(metrica_id, limiar, passo, abaixo, igual, acima, posicao) -> None:
    metrica = next(
        m for m in catalogo.carregar_catalogo(CATALOGO)["metricas"] if m["id"] == metrica_id
    )
    esperado = {-1: abaixo, 0: igual, 1: acima}[posicao]

    assert classificacao.situacao_medida(metrica, limiar + posicao * passo) == esperado


def test_leitor_observa_relatorio_completo_durante_atualizacao(tmp_path, monkeypatch) -> None:
    destino = tmp_path / "saude.json"
    anterior = {"resumo": {"criticos": [{"id": "modelo.roc_auc", "valor": 0.4}]}}
    novo = {"resumo": {"criticos": []}}
    destino.write_text(json.dumps(anterior), encoding="utf-8")
    dump_original = json.dump
    observados = []

    def escrever_com_leitura(conteudo, arquivo, **kwargs):
        # Uma leitura entre dois blocos da escrita deve ver o relatorio anterior.
        arquivo.write("{")
        arquivo.flush()
        observados.append(json.loads(destino.read_text(encoding="utf-8")))
        arquivo.seek(0)
        arquivo.truncate()
        dump_original(conteudo, arquivo, **kwargs)

    monkeypatch.setattr(json, "dump", escrever_com_leitura)
    resultados.gravar_saude(destino, novo)

    assert observados == [anterior]
    assert json.loads(destino.read_text(encoding="utf-8")) == novo
    assert list(tmp_path.iterdir()) == [destino]


def test_falha_no_meio_da_escrita_preserva_relatorio(tmp_path, monkeypatch) -> None:
    destino = tmp_path / "saude.json"
    anterior = '{"resumo": {"criticos": [{"id": "modelo.roc_auc"}]}}\n'
    destino.write_text(anterior, encoding="utf-8")

    def falhar(conteudo, arquivo, **kwargs):
        arquivo.write('{"resumo": ')
        arquivo.flush()
        raise OSError("Disco cheio durante a escrita")

    monkeypatch.setattr(json, "dump", falhar)
    with pytest.raises(OSError, match="Disco cheio"):
        resultados.gravar_saude(destino, {"resumo": {"criticos": []}})

    assert destino.read_text(encoding="utf-8") == anterior
    assert list(tmp_path.iterdir()) == [destino]


def test_falha_na_substituicao_preserva_relatorio(tmp_path, monkeypatch) -> None:
    destino = tmp_path / "saude.json"
    anterior = '{"resumo": {"criticos": []}}\n'
    destino.write_text(anterior, encoding="utf-8")

    def falhar(origem, alvo):
        raise OSError("Falha ao substituir relatorio")

    monkeypatch.setattr(arquivos.os, "replace", falhar)
    with pytest.raises(OSError, match="Falha ao substituir"):
        resultados.gravar_saude(destino, {"resumo": {"criticos": [{"id": "modelo.roc_auc"}]}})

    assert destino.read_text(encoding="utf-8") == anterior
    assert list(tmp_path.iterdir()) == [destino]
