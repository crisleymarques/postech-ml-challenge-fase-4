"""Treinamento e avaliacao do modelo baseline de Credit Scoring (issue F1-02).

Le as particoes de treino, validacao e teste aprovadas no manifesto da
validacao, confere hashes e schema efetivo dos artefatos, ajusta o pre-processamento e o
classificador apenas no treino, seleciona pela validacao e avalia no teste
depois da selecao. Persiste o pipeline, os metadados, a avaliacao e a
referencia congelada das features de treino.

Codigos de saida: 0 sucesso, 1 erro tecnico ou incompatibilidade de artefatos.

Uso:
    uv run python src/monitoramento/treinamento.py \
        --config configs/treinamento.json \
        --input-manifest reports/runs/<run_id>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Permite executar o arquivo diretamente na raiz do repositório, como no README.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from monitoramento.preprocessamento import converter_categoricas, converter_numericas

CONTRACT_VERSION = "1.0"
STAGE = "treinamento"
ETAPA_PRODUTORA = "validacao"

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

PARTICOES_ENTRADA = ("train", "validation", "test")
IDENTIFICADOR = "record_id"
LIMIAR = 0.5
METRICAS = ("accuracy", "precision", "recall", "f1", "roc_auc")
ALGORITMOS = ("logistic_regression",)


class ErroArtefato(Exception):
    """Indica dependencia ausente, divergente ou incompativel."""


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


def versoes_ambiente() -> dict[str, str]:
    """Registra as versoes relevantes do ambiente de execucao."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao da etapa."""
    if not caminho.is_file():
        raise ValueError(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = (
        "dataset_id",
        "schema_version",
        "model_id",
        "seed",
        "target",
        "features",
        "model",
        "outputs",
    )
    ausentes = [chave for chave in obrigatorios if chave not in config]
    if ausentes:
        raise ValueError(f"Chaves ausentes na configuracao: {ausentes}")

    if config["model"].get("algorithm") not in ALGORITMOS:
        raise ValueError(
            f"Algoritmo nao suportado: {config['model'].get('algorithm')}. "
            f"Esperado: {list(ALGORITMOS)}."
        )
    if config["model"].get("selection_metric") not in METRICAS:
        raise ValueError(f"Metrica de selecao invalida: {config['model'].get('selection_metric')}")
    return config


def carregar_manifesto(caminho: Path) -> dict[str, Any]:
    """Le o manifesto de entrada e exige o produtor com sucesso."""
    if not caminho.is_file():
        raise ValueError(f"Manifesto de entrada nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        manifesto = json.load(arquivo)

    if versao_maior(manifesto.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ValueError("Versao principal do contrato do manifesto incompativel.")
    if manifesto.get("stage") != ETAPA_PRODUTORA:
        raise ValueError(f"Etapa produtora inesperada: {manifesto.get('stage')}")
    if manifesto.get("status") != "success":
        raise ValueError(f"Dependencia sem sucesso: {manifesto.get('status')}")
    if manifesto.get("quality", {}).get("approved") is not True:
        raise ValueError("Manifesto de validacao sem aprovacao de qualidade.")
    if not manifesto.get("schema_version"):
        raise ValueError("Manifesto de validacao sem schema_version.")
    return manifesto


def resolver_particoes(
    manifesto: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Resolve particoes aprovadas, verificando hash e colunas do modelo."""
    artefatos = {
        item.get("artifact_type"): item
        for item in manifesto.get("inputs", [])
        if item.get("artifact_type") in PARTICOES_ENTRADA
    }
    faltantes = [nome for nome in PARTICOES_ENTRADA if nome not in artefatos]
    if faltantes:
        raise ErroArtefato(f"Particoes ausentes no manifesto: {faltantes}")

    for nome in PARTICOES_ENTRADA:
        artefato = artefatos[nome]
        caminho = Path(artefato["path"])
        if not caminho.is_file():
            raise ErroArtefato(f"Arquivo da particao ausente: {caminho}")
        if sha256_arquivo(caminho) != artefato.get("sha256"):
            raise ErroArtefato(f"Hash divergente para a particao {nome}.")
        colunas_arquivo = pd.read_csv(caminho, nrows=0).columns.tolist()
        colunas_manifesto = artefato.get("columns")
        if colunas_manifesto != colunas_arquivo:
            raise ErroArtefato(f"Schema efetivo divergente para a particao {nome}.")
        if config is not None:
            features = list(config["features"]["numeric"]) + list(
                config["features"]["categorical"]
            )
            necessarias = {*features, config["features"].get("identifier", IDENTIFICADOR)}
            necessarias.add(config["target"]["column"])
            colunas = set(colunas_arquivo)
            if not necessarias.issubset(colunas):
                raise ErroArtefato(
                    f"Schema da particao {nome} incompativel; colunas ausentes: "
                    f"{sorted(necessarias - colunas)}"
                )
    return artefatos


def validar_barreira(manifesto: dict[str, Any], config: dict[str, Any]) -> None:
    """Confere o vinculo da aprovacao com dataset, schema e entradas efetivas."""
    qualidade = manifesto.get("quality")
    if versao_maior(manifesto.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroArtefato("Versao principal do manifesto de validacao incompativel.")
    if manifesto.get("stage") != ETAPA_PRODUTORA:
        raise ErroArtefato(f"Etapa produtora inesperada: {manifesto.get('stage')}")
    if manifesto.get("status") != "success" or not isinstance(qualidade, dict):
        raise ErroArtefato("Treinamento exige manifesto de validacao concluido.")
    if qualidade.get("approved") is not True:
        raise ErroArtefato("Treinamento bloqueado: qualidade dos dados nao aprovada.")
    if manifesto.get("dataset_id") != config["dataset_id"]:
        raise ErroArtefato("dataset_id divergente entre configuracao e manifesto de validacao.")
    if qualidade.get("dataset_id") != config["dataset_id"]:
        raise ErroArtefato("dataset_id divergente no relatorio de qualidade.")
    schema_version = config["schema_version"]
    if not schema_version or manifesto.get("schema_version") != schema_version:
        raise ErroArtefato("schema_version divergente ou ausente no manifesto de validacao.")
    if qualidade.get("schema_version") != schema_version:
        raise ErroArtefato("schema_version divergente no relatorio de qualidade.")
    target_contract = manifesto.get("target_contract")
    target_esperado = config.get("target")
    if not isinstance(target_contract, dict):
        raise ErroArtefato("Manifesto de validacao sem contrato semantico do alvo.")
    campos_alvo = ("column", "positive_class", "negative_class", "positive_value", "negative_value")
    if not isinstance(target_esperado, dict) or any(
        target_contract.get(campo) != target_esperado.get(campo) for campo in campos_alvo
    ):
        raise ErroArtefato("Contrato semantico do alvo divergente entre validacao e treinamento.")
    por_tipo = {item.get("artifact_type"): item for item in qualidade.get("artifacts", [])}
    for item in manifesto.get("inputs", []):
        aprovado = por_tipo.get(item.get("artifact_type"))
        if (
            aprovado is None
            or aprovado.get("approved") is not True
            or aprovado.get("sha256") != item.get("sha256")
        ):
            raise ErroArtefato(
                f"Aprovacao de qualidade nao corresponde a entrada {item.get('artifact_type')}."
            )


def ler_particao(caminho: Path) -> pd.DataFrame:
    """Le uma particao como texto e trata campo vazio como ausencia."""
    base = pd.read_csv(caminho, dtype=str, na_filter=False)
    return base.replace("", np.nan)


def selecionar_features(config: dict[str, Any], base: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Valida e devolve as colunas numericas e categoricas do modelo."""
    especificacao = config["features"]
    numericas = list(especificacao["numeric"])
    categoricas = list(especificacao["categorical"])
    features = numericas + categoricas

    if not features:
        raise ValueError("Nenhuma feature definida na configuracao.")
    if len(set(features)) != len(features):
        raise ValueError("Features repetidas entre as listas numerica e categorica.")

    alvo = config["target"]["column"]
    identificador = especificacao.get("identifier", IDENTIFICADOR)
    tecnicas = {alvo, identificador}
    if tecnicas & set(features):
        raise ErroArtefato(
            f"Alvo ou identificador nao podem ser features: {tecnicas & set(features)}"
        )

    ausentes = [coluna for coluna in features if coluna not in base.columns]
    if ausentes:
        raise ErroArtefato(f"Features ausentes na particao: {ausentes}")
    return numericas, categoricas


def alvo_numerico(base: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Extrai o alvo binario validando o dominio esperado."""
    coluna = config["target"]["column"]
    if coluna not in base.columns:
        raise ErroArtefato(f"Coluna alvo ausente: {coluna}")

    valores_originais = pd.to_numeric(base[coluna], errors="raise")
    if valores_originais.isna().any() or not np.isfinite(valores_originais.to_numpy()).all():
        raise ErroArtefato("Rotulos de alvo ausentes ou nao finitos.")
    if not np.equal(valores_originais, np.floor(valores_originais)).all():
        raise ErroArtefato("Rotulos de alvo devem ser inteiros; truncamento nao e permitido.")
    if (config["target"]["positive_value"], config["target"]["negative_value"]) != (1, 0):
        raise ValueError("O treinamento suporta positive_value=1 e negative_value=0.")
    esperados = {0, 1}
    valores = valores_originais.astype(int)
    inesperados = sorted(set(valores.unique()) - esperados)
    if inesperados:
        raise ErroArtefato(f"Rotulos de alvo inesperados: {inesperados}")
    return valores


def construir_pipeline(
    numericas: list[str], categoricas: list[str], config: dict[str, Any]
) -> Pipeline:
    """Monta o pipeline com pre-processamento ajustavel e o classificador."""
    preprocessadores: list[tuple[str, Any, list[str]]] = []
    if numericas:
        preprocessadores.append(
            (
                "numerico",
                Pipeline(
                    [
                        ("converter", FunctionTransformer(converter_numericas)),
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                numericas,
            )
        )
    if categoricas:
        preprocessadores.append(
            (
                "categorico",
                Pipeline(
                    [
                        ("converter", FunctionTransformer(converter_categoricas)),
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categoricas,
            )
        )

    classificador = LogisticRegression(
        random_state=int(config["seed"]), **config["model"]["params"]
    )
    return Pipeline(
        [
            ("preprocessamento", ColumnTransformer(preprocessadores)),
            ("classificador", classificador),
        ]
    )


def calcular_assinatura(
    config: dict[str, Any],
    numericas: list[str],
    categoricas: list[str],
    hash_treino: str,
) -> str:
    """Deriva um identificador estavel do modelo a partir de suas entradas."""
    descritor = {
        "model_id": config["model_id"],
        "algorithm": config["model"]["algorithm"],
        "params": config["model"]["params"],
        "seed": config["seed"],
        "target": config["target"],
        "numeric": numericas,
        "categorical": categoricas,
        "train_sha256": hash_treino,
    }
    conteudo = json.dumps(descritor, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(conteudo).hexdigest()


def avaliar(
    modelo: Pipeline, base: pd.DataFrame, features: list[str], config: dict[str, Any]
) -> dict[str, float | None]:
    """Calcula as metricas de classificacao de uma particao."""
    alvo = alvo_numerico(base, config)
    probabilidade = modelo.predict_proba(base[features])[:, 1]
    predito = (probabilidade >= LIMIAR).astype(int)
    return {
        "accuracy": float(accuracy_score(alvo, predito)),
        "precision": float(precision_score(alvo, predito, zero_division=0)),
        "recall": float(recall_score(alvo, predito, zero_division=0)),
        "f1": float(f1_score(alvo, predito, zero_division=0)),
        "roc_auc": float(roc_auc_score(alvo, probabilidade)) if alvo.nunique() > 1 else None,
    }


def montar_avaliacoes(
    modelo: Pipeline,
    particoes: dict[str, pd.DataFrame],
    features: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Consolida as metricas de cada particao no formato do contrato."""
    avaliacoes: list[dict[str, Any]] = []
    for nome in PARTICOES_ENTRADA:
        base = particoes[nome]
        valores = avaliar(modelo, base, features, config)
        for metrica in METRICAS:
            avaliacoes.append(
                {
                    "name": metrica,
                    "value": valores[metrica],
                    "partition": nome,
                    "valid_labels": int(len(base)),
                }
            )
    return avaliacoes


def descritor_artefato(
    tipo: str,
    caminho: Path,
    run_id: str,
    linhas: int | None = None,
    colunas: list[str] | None = None,
) -> dict[str, Any]:
    """Descreve um artefato tabular conforme o manifesto comum."""
    artefato: dict[str, Any] = {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }
    if linhas is not None:
        artefato["row_count"] = int(linhas)
    if colunas is not None:
        artefato["columns"] = list(colunas)
    return artefato


def persistir_modelo(
    *,
    modelo: Pipeline,
    config: dict[str, Any],
    model_id: str,
    reference_id: str,
    numericas: list[str],
    categoricas: list[str],
    avaliacoes: list[dict[str, Any]],
    artefatos: dict[str, dict[str, Any]],
    run_id: str,
    inicio: str,
) -> list[dict[str, Any]]:
    """Grava pipeline, metadados e avaliacao e devolve os artefatos gerados."""
    features = numericas + categoricas
    diretorio = Path(config["outputs"]["models_dir"]) / model_id
    diretorio.mkdir(parents=True, exist_ok=False)

    joblib.dump(modelo, diretorio / "pipeline.pkl")

    metrica_selecao = config["model"]["selection_metric"]
    selecao = next(
        (
            item
            for item in avaliacoes
            if item["partition"] == "validation" and item["name"] == metrica_selecao
        ),
        None,
    )
    escrever_json(
        diretorio / "metadata.json",
        {
            "contract_version": CONTRACT_VERSION,
            "model_id": model_id,
            "reference_id": reference_id,
            "dataset_id": config["dataset_id"],
            "schema_version": config["schema_version"],
            "algorithm": config["model"]["algorithm"],
            "hyperparameters": config["model"]["params"],
            "target": config["target"],
            "features": features,
            "numeric_features": numericas,
            "categorical_features": categoricas,
            "prediction_threshold": LIMIAR,
            "score_meaning": (
                f"probabilidade estimada da classe {config['target']['positive_class']}"
            ),
            "seed": config["seed"],
            "created_at": inicio,
            "environment": versoes_ambiente(),
            "source_partitions": {
                nome: {
                    "path": artefatos[nome]["path"],
                    "sha256": artefatos[nome]["sha256"],
                    "producer_run_id": artefatos[nome].get("producer_run_id"),
                }
                for nome in PARTICOES_ENTRADA
            },
            "quality_validation_run_id": artefatos["train"].get("validated_by_run_id"),
        },
    )
    escrever_json(
        diretorio / "evaluation.json",
        {
            "model_id": model_id,
            "reference_id": reference_id,
            "positive_class": config["target"]["positive_class"],
            "prediction_threshold": LIMIAR,
            "selection": {
                "metric": metrica_selecao,
                "partition": "validation",
                "value": selecao["value"] if selecao else None,
            },
            "evaluations": avaliacoes,
        },
    )

    caminhos = (
        ("model_pipeline", diretorio / "pipeline.pkl"),
        ("model_metadata", diretorio / "metadata.json"),
        ("model_evaluation", diretorio / "evaluation.json"),
    )
    return [descritor_artefato(tipo, caminho, run_id) for tipo, caminho in caminhos]


def persistir_referencia(
    *,
    particao_treino: pd.DataFrame,
    config: dict[str, Any],
    reference_id: str,
    features: list[str],
    run_id: str,
) -> dict[str, Any]:
    """Congela as features de treino usadas como referencia de drift."""
    identificador = config["features"].get("identifier", IDENTIFICADOR)
    if identificador not in particao_treino.columns:
        raise ErroArtefato(f"Identificador ausente na particao de treino: {identificador}")

    diretorio = Path(config["outputs"]["reference_dir"]) / reference_id
    diretorio.mkdir(parents=True, exist_ok=False)
    conteudo = particao_treino[[identificador, *features]]
    caminho = diretorio / "features.csv"
    conteudo.to_csv(caminho, index=False, encoding="utf-8")
    return descritor_artefato("reference_features", caminho, run_id, len(conteudo), list(conteudo))


def registrar_evento(
    eventos: list[dict[str, Any]],
    run_id: str,
    dataset_id: str,
    evento: str,
    nivel: str,
    *,
    model_id: str | None = None,
    reference_id: str | None = None,
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
            "model_id": model_id,
            "reference_id": reference_id,
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
    status: str,
    iniciado: str,
    finalizado: str,
    config_path: Path,
    config: dict[str, Any],
    entradas: list[dict[str, Any]],
    saidas: list[dict[str, Any]],
    model_id: str | None,
    reference_id: str | None,
    error_code: str | None,
    selecao: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reune o manifesto comum da etapa de treinamento."""
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
        "schema_version": config.get("schema_version"),
        "model_id": model_id,
        "reference_id": reference_id,
        "scenario_id": None,
        "batch_id": None,
        "error_code": error_code,
        "algorithm": config["model"]["algorithm"],
        "hyperparameters": config["model"]["params"],
        "seed": config["seed"],
        "features": list(config["features"]["numeric"]) + list(config["features"]["categorical"]),
        "selection": selecao,
    }


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifesto: dict[str, Any],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa o treinamento completo e retorna o codigo de saida."""
    iniciado = agora_iso()
    dataset_id = config.get("dataset_id", "desconhecido")
    eventos: list[dict[str, Any]] = []
    registrar_evento(eventos, run_id, dataset_id, "started", "INFO")

    entradas: list[dict[str, Any]] = []
    saidas: list[dict[str, Any]] = []
    model_id: str | None = None
    reference_id: str | None = None
    selecao: dict[str, Any] | None = None

    try:
        validar_barreira(manifesto, config)

        artefatos = resolver_particoes(manifesto, config)
        for artefato in artefatos.values():
            artefato["validated_by_run_id"] = manifesto["run_id"]
        particoes = {
            nome: ler_particao(Path(artefatos[nome]["path"])) for nome in PARTICOES_ENTRADA
        }

        numericas, categoricas = selecionar_features(config, particoes["train"])
        features = numericas + categoricas

        assinatura = calcular_assinatura(
            config, numericas, categoricas, artefatos["train"]["sha256"]
        )
        # Cada execução publica artefatos imutáveis, mesmo quando os dados e a
        # configuração são idênticos. O sufixo também evita reuso da avaliação
        # quando apenas validação/teste mudam.
        sufixo_execucao = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
        model_id = f"{config['model_id']}-{assinatura[:8]}-{sufixo_execucao}"
        reference_id = f"ref-{assinatura[:8]}-{sufixo_execucao}"

        diretorio_modelo = Path(config["outputs"]["models_dir"]) / model_id
        diretorio_referencia = Path(config["outputs"]["reference_dir"]) / reference_id
        conflitos = [
            caminho.as_posix()
            for caminho in (diretorio_modelo, diretorio_referencia)
            if caminho.exists()
        ]
        if conflitos:
            raise ErroArtefato(f"Destinos imutaveis ja existem: {conflitos}")

        modelo = construir_pipeline(numericas, categoricas, config)
        modelo.fit(particoes["train"][features], alvo_numerico(particoes["train"], config))

        avaliacoes = montar_avaliacoes(modelo, particoes, features, config)
        selecao = next(
            (
                item
                for item in avaliacoes
                if item["partition"] == "validation"
                and item["name"] == config["model"]["selection_metric"]
            ),
            None,
        )

        saidas = persistir_modelo(
            modelo=modelo,
            config=config,
            model_id=model_id,
            reference_id=reference_id,
            numericas=numericas,
            categoricas=categoricas,
            avaliacoes=avaliacoes,
            artefatos=artefatos,
            run_id=run_id,
            inicio=iniciado,
        )
        saidas.append(
            persistir_referencia(
                particao_treino=particoes["train"],
                config=config,
                reference_id=reference_id,
                features=features,
                run_id=run_id,
            )
        )

        entradas = [
            {
                "artifact_type": nome,
                "path": artefatos[nome]["path"],
                "sha256": artefatos[nome]["sha256"],
                "producer_run_id": artefatos[nome].get("producer_run_id"),
                "validated_by_run_id": manifesto.get("run_id"),
                "row_count": artefatos[nome].get("row_count"),
                "columns": artefatos[nome].get("columns"),
            }
            for nome in PARTICOES_ENTRADA
        ]
        avaliacoes_metricas = avaliacoes
    except ErroArtefato as erro:
        registrar_evento(
            eventos, run_id, dataset_id, "failed", "ERROR", error_code="ARTEFATO_INCOMPATIVEL"
        )
        return finalizar(
            CODIGO_ERRO,
            "failed",
            "ARTEFATO_INCOMPATIVEL",
            str(erro),
            run_id,
            iniciado,
            config_path,
            config,
            eventos,
            entradas,
            saidas,
            model_id,
            reference_id,
            selecao,
            output_dir,
        )
    except Exception as erro:
        registrar_evento(eventos, run_id, dataset_id, "failed", "ERROR", error_code="ERRO_TECNICO")
        return finalizar(
            CODIGO_ERRO,
            "failed",
            "ERRO_TECNICO",
            str(erro),
            run_id,
            iniciado,
            config_path,
            config,
            eventos,
            entradas,
            saidas,
            model_id,
            reference_id,
            selecao,
            output_dir,
        )

    registrar_evento(
        eventos,
        run_id,
        dataset_id,
        "completed",
        "INFO",
        model_id=model_id,
        reference_id=reference_id,
    )
    return finalizar(
        CODIGO_SUCESSO,
        "success",
        None,
        None,
        run_id,
        iniciado,
        config_path,
        config,
        eventos,
        entradas,
        saidas,
        model_id,
        reference_id,
        selecao,
        output_dir,
        avaliacoes_metricas,
    )


def finalizar(
    codigo: int,
    status: str,
    error_code: str | None,
    mensagem: str | None,
    run_id: str,
    iniciado: str,
    config_path: Path,
    config: dict[str, Any],
    eventos: list[dict[str, Any]],
    entradas: list[dict[str, Any]],
    saidas: list[dict[str, Any]],
    model_id: str | None,
    reference_id: str | None,
    selecao: dict[str, Any] | None,
    output_dir: Path,
    avaliacoes: list[dict[str, Any]] | None = None,
) -> int:
    """Persiste config, eventos, metricas e manifesto e imprime o desfecho."""
    escrever_json(output_dir / "config.json", config)
    escrever_eventos(output_dir / "events.jsonl", eventos)

    medidas = [
        {
            "name": f"{item['name']}_{item['partition']}",
            "value": item["value"],
            "unit": "ratio",
            "availability": "available" if item["value"] is not None else "not_available",
        }
        for item in (avaliacoes or [])
        if item["partition"] in ("validation", "test")
        and item["name"] == config["model"]["selection_metric"]
    ]
    escrever_json(output_dir / "metrics.json", {"run_id": run_id, "measurements": medidas})

    manifesto = montar_manifesto(
        run_id=run_id,
        status=status,
        iniciado=iniciado,
        finalizado=agora_iso(),
        config_path=config_path,
        config=config,
        entradas=entradas,
        saidas=saidas,
        model_id=model_id,
        reference_id=reference_id,
        error_code=error_code,
        selecao=selecao,
    )
    escrever_json(output_dir / "manifest.json", manifesto)

    if status == "success":
        print(f"Treinamento concluido: {model_id}")
        if selecao and selecao["value"] is not None:
            print(f"  selecao ({selecao['name']}/{selecao['partition']}): {selecao['value']:.4f}")
        for artefato in saidas:
            print(f"  {artefato['artifact_type']:<18} -> {artefato['path']}")
        print(f"  manifesto          -> {output_dir.as_posix()}/manifest.json")
    else:
        print(f"Treinamento {status}: {mensagem}")
    return codigo


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e avalia o modelo baseline.")
    parser.add_argument("--config", type=Path, required=True, help="JSON de configuracao da etapa.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        required=True,
        help="Manifesto aprovado da validacao com particoes de treino, validacao e teste.",
    )
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    args = parser.parse_args()

    if len(args.input_manifest) != 1:
        parser.error("O treinamento consome exatamente um manifesto da validacao.")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id

    try:
        config = carregar_config(args.config)
        manifesto = carregar_manifesto(args.input_manifest[0])
    except Exception as erro:  # noqa: BLE001 - entrada invalida e erro tecnico
        print(f"Entrada invalida: {erro}")
        sys.exit(CODIGO_ERRO)

    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)
    output_dir.mkdir(parents=True)

    sys.exit(processar(args.config, config, manifesto, run_id, output_dir))


if __name__ == "__main__":
    main()
