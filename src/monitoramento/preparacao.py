"""Preparacao e separacao do dataset de referencia (issue F1-01).

Le a fonte autorizada indicada na configuracao, aplica transformacoes
deterministicas (sem aprender imputacao, escala, codificacao ou selecao de
variaveis a partir do conjunto completo) e separa quatro particoes disjuntas:
treino, validacao, teste e base de simulacao. Publica as particoes em CSV, o
manifesto de preparacao e as saidas de execucao (`config.json`, `events.jsonl`,
`metrics.json`) em `reports/runs/<run_id>/`.

Codigos de saida: 0 sucesso, 2 rejeicao de dados, 1 erro tecnico.

Uso:
    uv run python src/monitoramento/preparacao.py --config configs/preparacao.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONTRACT_VERSION = "1.0"
STAGE = "preparacao"

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1
CODIGO_REJEICAO = 2

PARTICOES = ("train", "validation", "test", "simulation_base")
ARQUIVO_PARTICAO = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
    "simulation_base": "simulation_base.csv",
}
# Diretorio de cada particao: referencia ou analise (producao simulada).
DESTINO_PARTICAO = {
    "train": "reference",
    "validation": "reference",
    "test": "reference",
    "simulation_base": "analysis",
}
IDENTIFICADOR = "record_id"


class RejeicaoDados(Exception):
    """Indica que a origem nao atende ao minimo para uma separacao segura."""


def agora_iso() -> str:
    """Retorna o instante atual em ISO 8601 UTC com sufixo Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_arquivo(caminho: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em blocos."""
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida a configuracao antes de iniciar efeitos em disco."""
    if not caminho.is_file():
        raise ValueError(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if not isinstance(config, dict):
        raise TypeError("A configuracao deve ser um objeto JSON.")
    obrigatorios = ("source", "target", "split", "outputs")
    ausentes = [chave for chave in obrigatorios if chave not in config]
    if ausentes:
        raise ValueError(f"Chaves ausentes na configuracao: {ausentes}")
    if config.get("contract_version") != CONTRACT_VERSION or config.get("stage") != STAGE:
        raise ValueError("contract_version/stage incompativeis com a etapa de preparacao.")
    for nome in obrigatorios:
        if not isinstance(config[nome], dict):
            raise TypeError(f"'{nome}' deve ser um objeto.")

    source = config["source"]
    if not isinstance(source.get("path"), str) or not source["path"].strip():
        raise ValueError("source.path deve ser um caminho nao vazio.")
    target = config["target"]
    for campo in ("column", "positive_class", "negative_class"):
        if not isinstance(target.get(campo), str) or not target[campo].strip():
            raise ValueError(f"target.{campo} deve ser texto nao vazio.")
    validar_mapeamento_alvo(target)

    split = config["split"]
    if split.get("strategy") != "temporal":
        raise ValueError("split.strategy deve ser 'temporal'.")
    for campo in ("time_column", "time_format"):
        if not isinstance(split.get(campo), str) or not split[campo].strip():
            raise ValueError(f"split.{campo} deve ser texto nao vazio.")
    if type(split.get("seed")) is not int:
        raise ValueError("split.seed deve ser um inteiro.")
    if "dataset_id" in config and (
        not isinstance(config["dataset_id"], str) or not config["dataset_id"].strip()
    ):
        raise ValueError("dataset_id deve ser texto nao vazio.")
    if "preparation_id" in config and (
        not isinstance(config["preparation_id"], str)
        or not config["preparation_id"].strip()
        or Path(config["preparation_id"]).name != config["preparation_id"]
        or config["preparation_id"] in (".", "..")
    ):
        raise ValueError("preparation_id deve ser um identificador de diretorio valido.")
    outputs = config["outputs"]
    for campo in ("reference_dir", "analysis_dir"):
        if not isinstance(outputs.get(campo), str) or not outputs[campo].strip():
            raise ValueError(f"outputs.{campo} deve ser caminho nao vazio.")
    if Path(outputs["reference_dir"]).resolve() == Path(outputs["analysis_dir"]).resolve():
        raise ValueError("outputs.reference_dir e outputs.analysis_dir devem ser distintos.")

    proporcoes = split.get("proportions")
    if not isinstance(proporcoes, dict):
        raise TypeError("split.proportions deve ser um objeto.")
    if set(proporcoes) != set(PARTICOES):
        raise ValueError(f"As proporcoes devem cobrir exatamente {list(PARTICOES)}.")
    if any(
        not isinstance(valor, numbers.Real)
        or isinstance(valor, bool)
        or not math.isfinite(valor)
        or valor <= 0
        for valor in proporcoes.values()
    ):
        raise ValueError("Todas as proporcoes devem ser finitas e positivas.")
    return config


def ler_origem(caminho: Path) -> pd.DataFrame:
    """Le a origem como texto, preservando os valores como espelhados."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Origem nao encontrada: {caminho}")
    return pd.read_csv(caminho, dtype=str, na_filter=False)


def limpar(base: pd.DataFrame, alvo: str) -> pd.DataFrame:
    """Aplica limpeza deterministica: remove espacos nas bordas dos textos."""
    limpa = base.copy()
    for coluna in limpa.columns:
        limpa[coluna] = limpa[coluna].str.strip()
    if IDENTIFICADOR not in limpa.columns:
        raise RejeicaoDados(f"Coluna obrigatoria ausente: {IDENTIFICADOR}")
    if alvo not in limpa.columns:
        raise RejeicaoDados(f"Coluna alvo ausente: {alvo}")
    if limpa[IDENTIFICADOR].eq("").any() or limpa[IDENTIFICADOR].duplicated().any():
        raise RejeicaoDados(f"{IDENTIFICADOR} deve ser unico e nao vazio.")
    return limpa


def binarizar_alvo(base: pd.DataFrame, target: dict[str, Any]) -> pd.DataFrame:
    """Substitui o alvo textual pela representacao binaria da classe positiva."""
    coluna = target["column"]
    validar_mapeamento_alvo(target)
    positivos = base[coluna] == target["positive_class"]
    negativos = base[coluna] == target["negative_class"]
    desconhecidos = sorted(set(base.loc[~positivos & ~negativos, coluna]))
    if desconhecidos:
        raise RejeicaoDados(f"Rotulos de alvo nao previstos: {desconhecidos}")

    binaria = base.copy()
    binaria[coluna] = np.where(
        positivos, int(target["positive_value"]), int(target["negative_value"])
    )
    return binaria


def validar_mapeamento_alvo(target: dict[str, Any]) -> None:
    """Garante duas classes textuais distintas e a codificacao binaria 0/1."""
    positivos = target.get("positive_class")
    negativos = target.get("negative_class")
    valor_positivo = target.get("positive_value")
    valor_negativo = target.get("negative_value")
    if (
        not isinstance(positivos, str)
        or not positivos.strip()
        or not isinstance(negativos, str)
        or not negativos.strip()
        or positivos == negativos
        or type(valor_positivo) is not int
        or type(valor_negativo) is not int
        or valor_positivo not in (0, 1)
        or valor_negativo not in (0, 1)
        or valor_positivo == valor_negativo
    ):
        raise ValueError(
            "Mapeamento do alvo exige classes textuais distintas e valores 0/1 distintos."
        )


def calcular_cotas(total: int, proporcoes: dict[str, float]) -> dict[str, int]:
    """Distribui `total` pelas particoes pelo metodo do maior resto."""
    soma = sum(proporcoes.values())
    exatas = {nome: total * valor / soma for nome, valor in proporcoes.items()}
    cotas = {nome: int(np.floor(valor)) for nome, valor in exatas.items()}
    faltantes = total - sum(cotas.values())
    ordem = sorted(exatas, key=lambda nome: exatas[nome] - cotas[nome], reverse=True)
    for nome in ordem[:faltantes]:
        cotas[nome] += 1
    return cotas


def separar_temporal(
    base: pd.DataFrame,
    coluna_tempo: str,
    formato_tempo: str,
    cotas: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """Separa em blocos temporais contiguos, do mais antigo ao mais recente."""
    datas = pd.to_datetime(base[coluna_tempo], format=formato_tempo, errors="coerce")
    if datas.isna().any():
        raise RejeicaoDados(
            f"Valores de {coluna_tempo} nao interpretaveis com o formato informado."
        )

    ordenada = (
        base.assign(_data=datas)
        .sort_values(["_data", IDENTIFICADOR], kind="stable")
        .drop(columns="_data")
        .reset_index(drop=True)
    )

    particoes: dict[str, pd.DataFrame] = {}
    inicio = 0
    for nome in PARTICOES:
        fim = inicio + cotas[nome]
        particoes[nome] = ordenada.iloc[inicio:fim].reset_index(drop=True)
        inicio = fim
    return particoes


def caminho_particao(config: dict[str, Any], preparation_id: str, nome: str) -> Path:
    """Resolve o caminho final de uma particao."""
    saidas = config["outputs"]
    raiz = (
        saidas["reference_dir"] if DESTINO_PARTICAO[nome] == "reference" else saidas["analysis_dir"]
    )
    return Path(raiz) / preparation_id / ARQUIVO_PARTICAO[nome]


def escrever_particoes(
    particoes: dict[str, pd.DataFrame],
    config: dict[str, Any],
    preparation_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Grava cada particao em diretorio novo e descreve o artefato gerado."""
    diretorios: list[Path] = []
    for nome in PARTICOES:
        diretorio = caminho_particao(config, preparation_id, nome).parent
        if diretorio not in diretorios:
            diretorios.append(diretorio)

    for diretorio in diretorios:
        if diretorio.exists():
            raise ValueError(f"Diretorio de saida ja existe: {diretorio}")
    for diretorio in diretorios:
        diretorio.mkdir(parents=True)

    artefatos: list[dict[str, Any]] = []
    for nome in PARTICOES:
        caminho = caminho_particao(config, preparation_id, nome)
        particoes[nome].to_csv(caminho, index=False, encoding="utf-8")
        artefatos.append(
            {
                "artifact_type": nome,
                "path": caminho.as_posix(),
                "sha256": sha256_arquivo(caminho),
                "producer_run_id": run_id,
                "row_count": int(len(particoes[nome])),
                "columns": list(particoes[nome].columns),
            }
        )
    return artefatos


def escrever_json(caminho: Path, conteudo: dict[str, Any]) -> None:
    """Grava um objeto JSON em UTF-8."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")


def registrar_evento(
    eventos: list[dict[str, Any]],
    run_id: str,
    dataset_id: str,
    evento: str,
    nivel: str,
    error_code: str | None = None,
) -> None:
    """Acrescenta um evento ao contrato comum de observabilidade."""
    eventos.append(
        {
            "timestamp": agora_iso(),
            "run_id": run_id,
            "stage": STAGE,
            "event": evento,
            "level": nivel,
            "dataset_id": dataset_id,
            "model_id": None,
            "reference_id": None,
            "scenario_id": None,
            "batch_id": None,
            "error_code": error_code,
        }
    )


def escrever_eventos(caminho: Path, eventos: list[dict[str, Any]]) -> None:
    """Grava os eventos em JSON Lines, um objeto por linha."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        for evento in eventos:
            arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def montar_manifesto(
    *,
    run_id: str,
    preparation_id: str,
    status: str,
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    entradas: list[dict[str, Any]],
    saidas: list[dict[str, Any]],
    error_code: str | None,
) -> dict[str, Any]:
    """Reune o manifesto comum da etapa de preparacao."""
    split = config["split"]
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": status,
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": entradas,
        "outputs": saidas,
        "dataset_id": config.get("dataset_id"),
        "schema_version": None,
        "model_id": None,
        "reference_id": None,
        "scenario_id": None,
        "batch_id": None,
        "error_code": error_code,
        "preparation_id": preparation_id,
        "source": {
            "url": config["source"].get("source_url"),
            "license": config["source"].get("license"),
        },
        "split": {
            "strategy": split["strategy"],
            "time_column": split["time_column"],
            "time_format": split["time_format"],
            "seed": split["seed"],
            "proportions": split["proportions"],
            "counts": {item["artifact_type"]: item["row_count"] for item in saidas},
        },
        "transformations": ["trim_espacos", "binarizar_alvo"],
        "roles": {
            nome: caminho_particao(config, preparation_id, nome).as_posix() for nome in PARTICOES
        },
    }


def processar(config_path: Path, config: dict[str, Any], run_id: str, output_dir: Path) -> int:
    """Executa a preparacao com config validada e retorna o codigo de saida."""
    preparacao_iniciada = agora_iso()
    dataset_id = config.get("dataset_id", "desconhecido")
    preparation_id = config.get("preparation_id", run_id)
    eventos: list[dict[str, Any]] = []
    entradas: list[dict[str, Any]] = []
    saidas: list[dict[str, Any]] = []

    registrar_evento(eventos, run_id, dataset_id, "started", "INFO")
    try:
        origem = Path(config["source"]["path"])
        alvo = config["target"]["column"]
        base = limpar(ler_origem(origem), alvo)
        base = binarizar_alvo(base, config["target"])

        if base.empty:
            raise RejeicaoDados("A origem nao contem registros de dados.")
        cotas = calcular_cotas(len(base), config["split"]["proportions"])
        vazias = [nome for nome, cota in cotas.items() if cota == 0]
        if vazias:
            raise RejeicaoDados(f"A origem e pequena demais; particoes sem registros: {vazias}.")
        particoes = separar_temporal(
            base,
            config["split"]["time_column"],
            config["split"]["time_format"],
            cotas,
        )

        entradas = [{
            "artifact_type": config["source"].get("artifact_type", "origem"),
            "path": origem.as_posix(),
            "sha256": sha256_arquivo(origem),
            "producer_run_id": None,
        }]
        saidas = escrever_particoes(particoes, config, preparation_id, run_id)
    except RejeicaoDados as erro:
        registrar_evento(eventos, run_id, dataset_id, "rejected", "ERROR", "REJEICAO_DADOS")
        return finalizar(
            CODIGO_REJEICAO,
            "rejected",
            "REJEICAO_DADOS",
            str(erro),
            run_id,
            preparation_id,
            preparacao_iniciada,
            config_path,
            config,
            eventos,
            entradas,
            output_dir,
        )
    except Exception as erro:
        registrar_evento(eventos, run_id, dataset_id, "failed", "ERROR", "ERRO_TECNICO")
        return finalizar(
            CODIGO_ERRO,
            "failed",
            "ERRO_TECNICO",
            str(erro),
            run_id,
            preparation_id,
            preparacao_iniciada,
            config_path,
            config,
            eventos,
            entradas,
            output_dir,
        )

    registrar_evento(eventos, run_id, dataset_id, "completed", "INFO")
    return finalizar(
        CODIGO_SUCESSO,
        "success",
        None,
        None,
        run_id,
        preparation_id,
        preparacao_iniciada,
        config_path,
        config,
        eventos,
        entradas,
        output_dir,
        saidas,
    )


def finalizar(
    codigo: int,
    status: str,
    error_code: str | None,
    mensagem: str | None,
    run_id: str,
    preparation_id: str,
    iniciado: str,
    config_path: Path,
    config: dict[str, Any],
    eventos: list[dict[str, Any]],
    entradas: list[dict[str, Any]],
    output_dir: Path,
    saidas: list[dict[str, Any]] | None = None,
) -> int:
    """Persiste config, eventos, metricas e manifesto e imprime o desfecho."""
    saidas = saidas or []
    escrever_json(output_dir / "config.json", config)
    escrever_eventos(output_dir / "events.jsonl", eventos)

    medidas = [
        {
            "name": f"linhas_{artefato['artifact_type']}",
            "value": artefato["row_count"],
            "unit": "linhas",
            "availability": "available",
        }
        for artefato in saidas
    ]
    escrever_json(output_dir / "metrics.json", {"run_id": run_id, "measurements": medidas})

    manifesto = montar_manifesto(
        run_id=run_id,
        preparation_id=preparation_id,
        status=status,
        iniciado=iniciado,
        finalizado=agora_iso(),
        config_path=config_path,
        config=config,
        entradas=entradas,
        saidas=saidas,
        error_code=error_code,
    )
    escrever_json(output_dir / "manifest.json", manifesto)

    if status == "success":
        print(f"Preparacao concluida: {preparation_id}")
        for artefato in saidas:
            tipo = artefato["artifact_type"]
            linhas = f"{artefato['row_count']:,}"
            print(f"  {tipo:<16} {linhas:>7} linhas -> {artefato['path']}")
        print(f"  manifesto        {output_dir.as_posix()}/manifest.json")
    else:
        print(f"Preparacao {status}: {mensagem}")
    return codigo


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara e separa o dataset de referencia.")
    parser.add_argument("--config", type=Path, required=True, help="JSON de configuracao da etapa.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        default=None,
        help="Nao aplicavel a preparacao; a origem vem de --config.",
    )
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    args = parser.parse_args()

    if args.input_manifest:
        parser.error("A preparacao nao consome manifesto de entrada; use --config.")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id

    try:
        config = carregar_config(args.config)
    except Exception as erro:  # noqa: BLE001 - configuracao invalida e erro tecnico
        print(f"Configuracao invalida: {erro}")
        sys.exit(CODIGO_ERRO)

    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)
    output_dir.mkdir(parents=True)

    sys.exit(processar(args.config, config, run_id, output_dir))


if __name__ == "__main__":
    main()

