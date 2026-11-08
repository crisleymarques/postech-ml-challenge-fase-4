import gzip
from unittest.mock import Mock

import pytest

from monitoramento import selecao_dataset as selecao

@pytest.fixture
def origem_padrao(tmp_path, monkeypatch):
    origem = tmp_path / "raw" / "accepted.csv"
    monkeypatch.setattr(selecao, "ORIGEM_PADRAO", origem)
    return origem


def test_origem_existente_dispensa_download(origem_padrao, monkeypatch):
    origem_padrao.parent.mkdir(parents=True)
    origem_padrao.write_text("arquivo local", encoding="utf-8")

    download = Mock()
    monkeypatch.setattr(selecao.kagglehub, "dataset_download", download)

    selecao.garantir_origem(origem_padrao)

    download.assert_not_called()
    assert origem_padrao.read_text(encoding="utf-8") == "arquivo local"


def test_primeira_execucao_baixa_e_segunda_reutiliza(
    origem_padrao, monkeypatch
):
    # Apenas um cabeçalho sintético; nenhum dado real.
    conteudo = ",".join(selecao.COLUNAS) + "\n"

    def download_simulado(handle, *, path, output_dir):
        destino = selecao.Path(output_dir)
        destino.mkdir(parents=True)
        compactado = destino / path

        with gzip.open(compactado, "wt", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)

        return str(compactado)

    download = Mock(side_effect=download_simulado)
    monkeypatch.setattr(selecao.kagglehub, "dataset_download", download)

    selecao.garantir_origem(origem_padrao)
    selecao.garantir_origem(origem_padrao)

    download.assert_called_once()
    assert origem_padrao.read_text(encoding="utf-8") == conteudo
    assert list(origem_padrao.parent.iterdir()) == [origem_padrao]


def test_falha_nao_deixa_origem_parcial(origem_padrao, monkeypatch):
    download = Mock(side_effect=ConnectionError("Falha simulada"))
    monkeypatch.setattr(selecao.kagglehub, "dataset_download", download)

    with pytest.raises(RuntimeError, match="Não foi possível preparar"):
        selecao.garantir_origem(origem_padrao)

    assert not origem_padrao.exists()
    assert list(origem_padrao.parent.iterdir()) == []


def test_origem_personalizada_ausente_nao_baixa(
    origem_padrao, tmp_path, monkeypatch
):
    download = Mock()
    monkeypatch.setattr(selecao.kagglehub, "dataset_download", download)

    with pytest.raises(FileNotFoundError, match="personalizada"):
        selecao.garantir_origem(tmp_path / "outra_origem.csv")

    download.assert_not_called()