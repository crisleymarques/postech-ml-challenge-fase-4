"""Simulacao temporal de Data Drift da producao (issue F2-01).

Le a base reservada para producao simulada, avanca o periodo de emissao para
representar a passagem do tempo e aplica mudancas economicas deterministas que
alteram a distribuicao de pelo menos duas features importantes do modelo. A
referencia original nao e reescrita: cada execucao grava um conjunto novo e
publica o manifesto do produtor (`stage: simulacao`) com o periodo, as
variaveis alteradas, as transformacoes e os parametros efetivos. Os rotulos,
quando existem, sao separados em arquivo proprio.

Codigos de saida: 0 sucesso, 1 erro tecnico.

Uso:
    uv run python src/monitoramento/simulacao_drift.py \
        --config configs/simulacao.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONTRACT_VERSION = "1.0"
STAGE = "simulacao"
IDENTIFICADOR = "record_id"

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

OPERACOES = ("scale", "shift")
MINIMO_VARIAVEIS = 2


class ErroSimulacao(Exception):
    """Indica entrada invalida para a geracao do cenario simulado."""


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


def escrever_json(caminho: Path, conteudo: dict[str, Any]) -> None:
    """Grava um objeto JSON em UTF-8."""
    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")


def versao_maior(versao: Any) -> str:
    """Extrai o componente principal de uma versao de contrato."""
    return str(versao or "").split(".")[0]


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao da simulacao."""
    if not caminho.is_file():
        raise ErroSimulacao(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = (
        "dataset_id",
        "schema_version",
        "scenario_id",
        "base",
        "time",
        "target",
        "drift",
        "outputs",
    )
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ValueError(f"Chaves ausentes na configuracao: {ausentes}")

    variaveis = config["drift"].get("variables", {})
    if len(variaveis) < MINIMO_VARIAVEIS:
        raise ValueError(
            f"A simulacao de Data Drift exige ao menos {MINIMO_VARIAVEIS} variaveis alteradas."
        )
    for coluna, especificacao in variaveis.items():
        operacao = especificacao.get("operation")
        if operacao not in OPERACOES:
            raise ValueError(
                f"Operacao invalida para {coluna}: {operacao}. Esperado: {list(OPERACOES)}."
            )
        if operacao == "scale" and "factor" not in especificacao:
            raise ValueError(f"Operacao de escala exige 'factor' em {coluna}.")
        if operacao == "shift" and "delta" not in especificacao:
            raise ValueError(f"Operacao de deslocamento exige 'delta' em {coluna}.")
        if len(especificacao.get("clip", [])) != 2:
            raise ValueError(f"Limite de faixa invalido para {coluna}.")
    return config


def carregar_base(caminho: Path) -> pd.DataFrame:
    """Le a base de origem mantendo os valores como texto."""
    if not caminho.is_file():
        raise ErroSimulacao(f"Base de origem nao encontrada: {caminho}")
    return pd.read_csv(caminho, dtype=str, na_filter=False)


def resumo_periodo(serie: pd.Series, formato: str) -> dict[str, str]:
    """Descreve o menor e o maior periodo de uma coluna temporal."""
    datas = pd.to_datetime(serie, format=formato, errors="raise")
    return {"min": datas.min().strftime(formato), "max": datas.max().strftime(formato)}


def deslocar_periodo(serie: pd.Series, formato: str, meses: int) -> pd.Series:
    """Avanca a coluna temporal em um numero fixo de meses."""
    datas = pd.to_datetime(serie, format=formato, errors="raise")
    return (datas + pd.DateOffset(months=int(meses))).dt.strftime(formato)


def aplicar_drift(
    base: pd.DataFrame,
    variaveis: dict[str, Any],
    seed: int,
    jitter: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Altera a distribuicao das variaveis escolhidas e descreve cada mudanca."""
    ajustado = base.copy()
    gerador = np.random.default_rng(seed)
    descritas: list[dict[str, Any]] = []
    for coluna, especificacao in variaveis.items():
        if coluna not in ajustado.columns:
            raise ErroSimulacao(f"Coluna da alteracao ausente na base: {coluna}")

        valores = pd.to_numeric(ajustado[coluna], errors="coerce")
        ausentes = valores.isna()

        ruido = gerador.uniform(1 - jitter, 1 + jitter, size=len(valores))
        operacao = especificacao["operation"]
        if operacao == "scale":
            parametro = float(especificacao["factor"])
            transformados = valores * parametro * ruido
        else:
            parametro = float(especificacao["delta"])
            transformados = valores + parametro * ruido

        minimo, maximo = especificacao["clip"]
        resultado = transformados.clip(float(minimo), float(maximo)).astype(object)
        # Ausencias permanecem ausentes: a simulacao nao inventa valor onde nao havia.
        resultado[ausentes] = ""
        ajustado[coluna] = resultado
        descritas.append(
            {
                "column": coluna,
                "operation": operacao,
                "parameter": parametro,
                "clip": [float(minimo), float(maximo)],
                "missing_preserved": int(ausentes.sum()),
                "mean_before": round(float(valores.mean()), 4),
                "mean_after": round(float(pd.to_numeric(ajustado[coluna]).mean()), 4),
            }
        )
    return ajustado, descritas


def separar_features_rotulos(base: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa as features dos rotulos, mantendo o identificador em ambos."""
    if alvo not in base.columns:
        raise ErroSimulacao(f"Coluna alvo ausente na base: {alvo}")
    if IDENTIFICADOR not in base.columns:
        raise ErroSimulacao(f"Coluna obrigatoria ausente: {IDENTIFICADOR}")

    features = base.drop(columns=[alvo])
    rotulos = base[[IDENTIFICADOR, alvo]]
    return features, rotulos


def escrever_artefato(caminho: Path, base: pd.DataFrame) -> None:
    """Grava um artefato tabular em CSV preservando as colunas observadas."""
    base.to_csv(caminho, index=False, encoding="utf-8", mode="x")


def descritor_artefato(tipo: str, caminho: Path, run_id: str, base: pd.DataFrame) -> dict[str, Any]:
    """Descreve um artefato conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
        "row_count": int(len(base)),
        "columns": list(base.columns),
    }


def registrar_evento(
    eventos: list[dict[str, Any]],
    run_id: str,
    dataset_id: str,
    scenario_id: str,
    batch_id: str,
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
            "scenario_id": scenario_id,
            "batch_id": batch_id,
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
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    batch_id: str,
    seed: int,
    entrada: dict[str, Any],
    saidas: list[dict[str, Any]],
    periodo: dict[str, Any],
    descritas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor do cenario simulado."""
    variaveis = sorted(descritas, key=lambda item: item["column"])
    transformacoes = [
        (f"deslocamento temporal de {periodo['column']} em {periodo['shift_months']} meses"),
        *(
            f"{item['operation']} em {item['column']} (parametro {item['parameter']})"
            for item in variaveis
        ),
    ]
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "success",
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": [entrada],
        "outputs": saidas,
        "dataset_id": config.get("dataset_id"),
        "schema_version": config.get("schema_version"),
        "model_id": None,
        "reference_id": None,
        "scenario_id": config.get("scenario_id"),
        "batch_id": batch_id,
        "seed": seed,
        "periodo": periodo,
        "variaveis_alteradas": [item["column"] for item in variaveis],
        "transformacoes": transformacoes,
        "parametros": {
            "jitter": float(config["drift"].get("jitter", 0.0)),
            "shift_months": int(periodo["shift_months"]),
            "variables": {item["column"]: item for item in variaveis},
        },
        "error_code": None,
    }


def montar_medidas(
    saidas: list[dict[str, Any]], descritas: list[dict[str, Any]], shift: int
) -> list:
    """Converte a simulacao gerada em medidas do contrato de metricas."""
    linhas = sum(item["row_count"] for item in saidas if item["artifact_type"] == "features")
    return [
        {
            "name": "linhas_geradas",
            "value": int(linhas),
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "variaveis_alteradas",
            "value": len(descritas),
            "unit": "variaveis",
            "availability": "available",
        },
        {
            "name": "deslocamento_meses",
            "value": int(shift),
            "unit": "meses",
            "availability": "available",
        },
    ]


def gerar(
    *,
    config_path: Path,
    source_path: Path | None,
    analysis_dir: Path | None,
    batch_id: str,
    run_id: str,
    output_dir: Path,
    seed: int | None,
) -> int:
    """Gera o cenario de Data Drift e publica manifesto, eventos e metricas."""
    try:
        config = carregar_config(config_path)
        dataset_id = config["dataset_id"]
        scenario_id = config["scenario_id"]
        destino = (analysis_dir or Path(config["outputs"]["analysis_dir"])) / scenario_id / batch_id
        semente = int(seed if seed is not None else config["drift"]["seed"])
    except (ErroSimulacao, ValueError, TypeError, KeyError, OSError) as erro:
        print(f"Configuracao invalida ({type(erro).__name__}).", file=sys.stderr)
        return CODIGO_ERRO

    # A verificacao antecipa colisoes comuns; mkdir exclusivo protege tambem corridas.
    if output_dir.exists() or destino.exists():
        print("Diretorio de execucao ou lote ja existe.", file=sys.stderr)
        return CODIGO_ERRO
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as erro:
        print(f"Falha ao reservar execucao ({type(erro).__name__}).", file=sys.stderr)
        return CODIGO_ERRO

    iniciado = agora_iso()
    eventos: list[dict[str, Any]] = []
    try:
        destino.mkdir(parents=True, exist_ok=False)
        registrar_evento(eventos, run_id, dataset_id, scenario_id, batch_id, "started", "INFO")

        origem = source_path or Path(config["base"]["path"])
        base = carregar_base(origem)

        tempo = config["time"]
        coluna_tempo = tempo["column"]
        formato = tempo["format"]
        if coluna_tempo not in base.columns:
            raise ErroSimulacao(f"Coluna temporal ausente na base: {coluna_tempo}")

        jitter = float(config["drift"].get("jitter", 0.0))

        periodo_origem = resumo_periodo(base[coluna_tempo], formato)
        base_simulada = base.copy()
        base_simulada[coluna_tempo] = deslocar_periodo(
            base[coluna_tempo], formato, tempo["shift_months"]
        )
        base_simulada, descritas = aplicar_drift(
            base_simulada, config["drift"]["variables"], semente, jitter
        )
        periodo_simulado = resumo_periodo(base_simulada[coluna_tempo], formato)

        alvo = config["target"]["column"]
        features, rotulos = separar_features_rotulos(base_simulada, alvo)

        caminho_features = destino / "features.csv"
        caminho_rotulos = destino / "labels.csv"
        escrever_artefato(caminho_features, features)
        escrever_artefato(caminho_rotulos, rotulos)

        entrada = {
            "artifact_type": config["base"].get("artifact_type", "origem"),
            "path": origem.as_posix(),
            "sha256": sha256_arquivo(origem),
            "row_count": int(len(base)),
            "columns": list(base.columns),
        }
        saidas = [
            descritor_artefato("features", caminho_features, run_id, features),
            descritor_artefato("labels", caminho_rotulos, run_id, rotulos),
        ]

        periodo = {
            "column": coluna_tempo,
            "format": formato,
            "origem": periodo_origem,
            "simulado": periodo_simulado,
            "shift_months": int(tempo["shift_months"]),
        }

        escrever_json(output_dir / "config.json", config)
        escrever_eventos(output_dir / "events.jsonl", eventos)
        escrever_json(
            output_dir / "metrics.json",
            {
                "run_id": run_id,
                "measurements": montar_medidas(saidas, descritas, periodo["shift_months"]),
            },
        )
        manifesto = montar_manifesto(
            run_id=run_id,
            iniciado=iniciado,
            finalizado=agora_iso(),
            config_path=config_path,
            config=config,
            batch_id=batch_id,
            seed=semente,
            entrada=entrada,
            saidas=saidas,
            periodo=periodo,
            descritas=descritas,
        )
        escrever_json(output_dir / "manifest.json", manifesto)
        registrar_evento(eventos, run_id, dataset_id, scenario_id, batch_id, "completed", "INFO")
        escrever_eventos(output_dir / "events.jsonl", eventos)
    except (ErroSimulacao, ValueError, OSError) as erro:
        print(f"Falha ao gerar o cenario simulado: {erro}")
        return CODIGO_ERRO

    print(f"Cenario simulado ({scenario_id}): {len(features)} linhas")
    print(
        f"  periodo {periodo['origem']['min']}..{periodo['origem']['max']} ->"
        f" {periodo['simulado']['min']}..{periodo['simulado']['max']}"
    )
    for item in descritas:
        print(
            f"    {item['column']:<16} {item['operation']:<6} "
            f"media {item['mean_before']} -> {item['mean_after']}"
        )
    print(f"  features -> {caminho_features.as_posix()}")
    print(f"  labels   -> {caminho_rotulos.as_posix()}")
    print(f"  manifesto -> {output_dir.as_posix()}/manifest.json")
    return CODIGO_SUCESSO


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula a passagem do tempo com Data Drift.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/simulacao.json"), help="JSON da etapa."
    )
    parser.add_argument(
        "--source", type=Path, default=None, help="Base de origem (sobrepoe a configuracao)."
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=None, help="Diretorio raiz do conjunto simulado."
    )
    parser.add_argument("--batch-id", default=None, help="Identificador do lote simulado.")
    parser.add_argument("--seed", type=int, default=None, help="Semente da simulacao.")
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    batch_id = args.batch_id or f"batch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    output_dir = args.output_dir or Path("reports/runs") / run_id

    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(
        gerar(
            config_path=args.config,
            source_path=args.source,
            analysis_dir=args.analysis_dir,
            batch_id=batch_id,
            run_id=run_id,
            output_dir=output_dir,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
