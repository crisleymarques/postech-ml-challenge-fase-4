"""Simulacao de Concept Drift da producao (issue F2-05).

Le a base reservada para producao simulada e altera a relacao entre o perfil e
o desfecho para um segmento definido: em um cenario de recessao, tomadores de um
segmento passam a inadimplir mais do que o historico projetava. As features
permanecem inalteradas; apenas o desfecho muda. Como a entrada e o mesmo perfil
passam a ter outro rotulo, trata-se de mudanca de conceito (P(y|X)) e nao de
mudanca de distribuicao (P(X)), que e o Data Drift.

A base original nao e reescrita: cada execucao grava as features inalteradas, os
rotulos do novo conceito, os rotulos originais (contraste) e publica o manifesto
do produtor (`stage: simulacao`) com o segmento, a regra e os parametros
efetivos. Somente rotulos negativos podem virar positivos: a recessao adiciona
inadimplencia, nao desfaz inadimplencia observada.

Codigos de saida: 0 sucesso, 1 erro tecnico.

Uso:
    uv run python src/monitoramento/simulacao_concept_drift.py \
        --config configs/simulacao_concept.json
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

OPERADORES = (">=", ">", "<=", "<")
OPERACAO = "raise_default_probability"


class ErroSimulacao(Exception):
    """Indica entrada invalida para a geracao do cenario de Concept Drift."""


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


def numero(valor: Any) -> bool:
    """Indica se o valor e um numero utilizavel (exclui booleanos)."""
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao do cenario de Concept Drift."""
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
        "target",
        "concept",
        "outputs",
    )
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ValueError(f"Chaves ausentes na configuracao: {ausentes}")

    alvo = config["target"]
    for chave in ("column", "positive_value", "negative_value"):
        if chave not in alvo:
            raise ValueError(f"O alvo exige a chave '{chave}'.")

    conceito = config["concept"]
    if not conceito.get("rationale"):
        raise ValueError("O cenario de Concept Drift exige 'rationale'.")
    segmento = conceito.get("segment", {})
    if not segmento.get("column"):
        raise ValueError("O segmento exige a chave 'column'.")
    if segmento.get("operator") not in OPERADORES:
        raise ValueError(f"Operador de segmento invalido: {segmento.get('operator')}")
    if not numero(segmento.get("value")):
        raise ValueError("O segmento exige 'value' numerico.")

    perigo = conceito.get("hazard", {})
    for chave in ("reference_value", "slope", "scale"):
        if not numero(perigo.get(chave)):
            raise ValueError(f"O hazard exige '{chave}' numerico.")
    if not numero(perigo.get("max_probability")) or not 0 < float(perigo["max_probability"]) <= 1:
        raise ValueError("'max_probability' deve estar entre 0 (exclusivo) e 1.")
    if float(perigo["scale"]) <= 0:
        raise ValueError("'scale' deve ser positivo.")
    return config


def carregar_base(caminho: Path) -> pd.DataFrame:
    """Le a base de origem mantendo os valores como texto."""
    if not caminho.is_file():
        raise ErroSimulacao(f"Base de origem nao encontrada: {caminho}")
    return pd.read_csv(caminho, dtype=str, na_filter=False)


def calcular_perigo(valores: pd.Series, mascara: np.ndarray, perigo: dict[str, Any]) -> np.ndarray:
    """Calcula a probabilidade extra de inadimplencia por registro do segmento.

    A probabilidade extra cresce com o excedente da variavel de risco em relacao
    ao valor de referencia, limitada em zero e no maximo declarado. Fora do
    segmento a probabilidade extra e zero.
    """
    referencia = float(perigo["reference_value"])
    inclinacao = float(perigo["slope"])
    escala = float(perigo["scale"])
    maximo = float(perigo["max_probability"])

    excedente = (valores - referencia).clip(lower=0.0) / escala
    bruto = (inclinacao * excedente).clip(lower=0.0, upper=maximo)

    return bruto.where(pd.Series(mascara, index=valores.index), 0.0).fillna(0.0).to_numpy()


def mascara_segmento(valores: pd.Series, segmento: dict[str, Any]) -> np.ndarray:
    """Devolve a mascara booleana dos registros pertencentes ao segmento."""
    limite = float(segmento["value"])
    operador = segmento["operator"]
    if operador == ">=":
        mascara = valores >= limite
    elif operador == ">":
        mascara = valores > limite
    elif operador == "<=":
        mascara = valores <= limite
    else:
        mascara = valores < limite
    return mascara.fillna(False).to_numpy()


def aplicar_mudanca_conceito(
    base: pd.DataFrame,
    alvo: str,
    positivo: int,
    conceito: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Altera a relacao perfil-desfecho no segmento e descreve a mudanca.

    Somente rotulos negativos dentro do segmento podem virar positivos, com
    probabilidade dada pela regra de perigo. A escolha dos registros usa uma
    semente fixa para ser reproduzivel.
    """
    segmento = conceito["segment"]
    coluna = segmento["column"]
    if coluna not in base.columns:
        raise ErroSimulacao(f"Coluna do segmento ausente na base: {coluna}")
    if alvo not in base.columns:
        raise ErroSimulacao(f"Coluna alvo ausente na base: {alvo}")
    if IDENTIFICADOR not in base.columns:
        raise ErroSimulacao(f"Coluna obrigatoria ausente: {IDENTIFICADOR}")

    rotulos = pd.to_numeric(base[alvo], errors="coerce")
    if rotulos.isna().any():
        raise ErroSimulacao("A base contem rotulos ausentes ou nao numericos.")

    positivos = int(positivo)
    valores = pd.to_numeric(base[coluna], errors="coerce")
    mascara = mascara_segmento(valores, segmento)
    perigo = calcular_perigo(valores, mascara, conceito["hazard"])

    gerador = np.random.default_rng(seed)
    sorteio = gerador.uniform(0.0, 1.0, size=len(base))

    elegiveis = (rotulos.to_numpy() != positivos) & mascara
    vira = elegiveis & (sorteio < perigo)

    novos = rotulos.to_numpy().astype(int).copy()
    novos[vira] = positivos

    alterada = base.copy()
    alterada[alvo] = novos.astype(str)

    taxa_antes = float((rotulos.to_numpy() == positivos).mean())
    taxa_depois = float((novos == positivos).mean())
    descricao = {
        "description": conceito["rationale"],
        "segment": {
            "column": coluna,
            "operator": segmento["operator"],
            "value": float(segmento["value"]),
        },
        "operation": OPERACAO,
        "parameters": {
            "reference_value": float(conceito["hazard"]["reference_value"]),
            "slope": float(conceito["hazard"]["slope"]),
            "scale": float(conceito["hazard"]["scale"]),
            "max_probability": float(conceito["hazard"]["max_probability"]),
            "positive_value": positivos,
        },
        "segment_rows": int(mascara.sum()),
        "flipped_rows": int(vira.sum()),
        "positive_rate_before": round(taxa_antes, 6),
        "positive_rate_after": round(taxa_depois, 6),
    }
    return alterada, descricao


def separar_features_rotulos(base: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa as features dos rotulos, mantendo o identificador em ambos."""
    features = base.drop(columns=[alvo])
    rotulos = base[[IDENTIFICADOR, alvo]]
    return features, rotulos


def escrever_artefato(caminho: Path, base: pd.DataFrame) -> None:
    """Grava um artefato tabular em CSV preservando as colunas observadas."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(caminho, index=False, encoding="utf-8")


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


def montar_medidas(descricao: dict[str, Any], linhas: int) -> list[dict[str, Any]]:
    """Converte o cenario gerado em medidas do contrato de metricas."""
    return [
        {
            "name": "linhas_geradas",
            "value": int(linhas),
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "registros_no_segmento",
            "value": descricao["segment_rows"],
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "rotulos_alterados",
            "value": descricao["flipped_rows"],
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "taxa_positivos_antes",
            "value": descricao["positive_rate_before"],
            "unit": "proporcao",
            "availability": "available",
        },
        {
            "name": "taxa_positivos_depois",
            "value": descricao["positive_rate_after"],
            "unit": "proporcao",
            "availability": "available",
        },
    ]


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
    descricao: dict[str, Any],
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor do cenario de Concept Drift."""
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
        "drift_type": "concept",
        "features_unchanged": True,
        "concept_change": descricao,
        "transformacoes": [
            f"mudanca de conceito em P(y|X): {OPERACAO} no segmento "
            f"{descricao['segment']['column']} {descricao['segment']['operator']} "
            f"{descricao['segment']['value']}"
        ],
        "error_code": None,
    }


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
    """Gera o cenario de Concept Drift e publica manifesto, eventos e metricas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    iniciado = agora_iso()
    eventos: list[dict[str, Any]] = []

    try:
        config = carregar_config(config_path)
        dataset_id = config["dataset_id"]
        scenario_id = config["scenario_id"]
        registrar_evento(eventos, run_id, dataset_id, scenario_id, batch_id, "started", "INFO")

        origem = source_path or Path(config["base"]["path"])
        base = carregar_base(origem)

        alvo = config["target"]["column"]
        positivos = int(config["target"]["positive_value"])
        semente = int(seed if seed is not None else config["concept"].get("seed", 42))

        alterada, descricao = aplicar_mudanca_conceito(
            base, alvo, positivos, config["concept"], semente
        )

        features, rotulos_ref = separar_features_rotulos(base, alvo)
        _, rotulos_novos = separar_features_rotulos(alterada, alvo)

        destino = (analysis_dir or Path(config["outputs"]["analysis_dir"])) / scenario_id
        caminho_features = destino / batch_id / "features.csv"
        caminho_rotulos = destino / batch_id / "labels.csv"
        caminho_ref = destino / batch_id / "labels_reference.csv"
        escrever_artefato(caminho_features, features)
        escrever_artefato(caminho_rotulos, rotulos_novos)
        escrever_artefato(caminho_ref, rotulos_ref)

        entrada = {
            "artifact_type": config["base"].get("artifact_type", "origem"),
            "path": origem.as_posix(),
            "sha256": sha256_arquivo(origem),
            "row_count": int(len(base)),
            "columns": list(base.columns),
        }
        saidas = [
            descritor_artefato("features", caminho_features, run_id, features),
            descritor_artefato("labels", caminho_rotulos, run_id, rotulos_novos),
            descritor_artefato("labels_reference", caminho_ref, run_id, rotulos_ref),
        ]

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {"run_id": run_id, "measurements": montar_medidas(descricao, len(features))},
        )
        escrever_json(
            output_dir / "manifest.json",
            montar_manifesto(
                run_id=run_id,
                iniciado=iniciado,
                finalizado=agora_iso(),
                config_path=config_path,
                config=config,
                batch_id=batch_id,
                seed=semente,
                entrada=entrada,
                saidas=saidas,
                descricao=descricao,
            ),
        )
        registrar_evento(eventos, run_id, dataset_id, scenario_id, batch_id, "completed", "INFO")
        escrever_eventos(output_dir / "events.jsonl", eventos)
    except (ErroSimulacao, ValueError) as erro:
        print(f"Falha ao gerar o cenario de Concept Drift: {erro}")
        return CODIGO_ERRO

    print(f"Cenario de Concept Drift ({scenario_id}): {len(features)} linhas")
    print(
        f"  segmento {descricao['segment']['column']} {descricao['segment']['operator']} "
        f"{descricao['segment']['value']}: {descricao['segment_rows']} registros"
    )
    print(
        f"  rotulos alterados: {descricao['flipped_rows']} "
        f"(taxa de positivos {descricao['positive_rate_before']} -> "
        f"{descricao['positive_rate_after']})"
    )
    print(f"  features -> {caminho_features.as_posix()}")
    print(f"  labels   -> {caminho_rotulos.as_posix()}")
    print(f"  manifesto -> {output_dir.as_posix()}/manifest.json")
    return CODIGO_SUCESSO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simula mudanca de conceito (Concept Drift) na producao."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulacao_concept.json"),
        help="JSON da etapa.",
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
