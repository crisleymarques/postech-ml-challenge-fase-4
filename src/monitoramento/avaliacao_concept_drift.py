"""Avaliacao do desempenho sob Concept Drift (issue F2-05).

Le o manifesto do treinamento (modelo e referencia) e o manifesto do cenario de
Concept Drift (features inalteradas, rotulos do novo conceito e rotulos
originais), confere contrato, `dataset_id` e hash dos artefatos, aplica o
pipeline persistido sobre as features e compara o desempenho do modelo entre os
dois conceitos.

Como as features e os registros sao os mesmos, apenas a relacao entre perfil e
desfecho muda: o contraste com os rotulos originais isola o efeito do Concept
Drift (P(y|X)) de uma mudanca de distribuicao das entradas (P(X)). O documento
`performance.json` registra as metricas de cada conceito, as variacoes, o
criterio de degradacao declarado e as conclusoes, deixando explicita a diferenca
entre Data Drift, Concept Drift e queda de desempenho. Um alerta nao e erro: a
execucao termina em `success` e emite `drift_alert`.

Codigos de saida: 0 sucesso, 1 erro tecnico ou incompatibilidade de artefatos.

Uso:
    uv run python src/monitoramento/avaliacao_concept_drift.py \
        --config configs/avaliacao_concept.json \
        --input-manifest reports/runs/<treinamento>/manifest.json \
        --input-manifest reports/runs/<concept-drift>/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CONTRACT_VERSION = "1.0"
STAGE = "avaliacao"
IDENTIFICADOR = "record_id"

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

METRICAS = ("accuracy", "precision", "recall", "f1", "roc_auc")

# Conversores de pre-processamento do treinamento que podem ter sido
# serializados no namespace `__main__` quando a etapa roda como script.
CONVERSORES_DO_TREINO = ("converter_numericas", "converter_categoricas")


class ErroAvaliacao(Exception):
    """Indica dependencia ausente, divergente ou incompativel."""


class ErroConfiguracao(Exception):
    """Indica configuracao ausente ou invalida."""


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
    """Le e valida minimamente a configuracao da avaliacao."""
    if not caminho.is_file():
        raise ErroConfiguracao(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroConfiguracao("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "model", "batch", "metrics")
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ErroConfiguracao(f"Chaves ausentes na configuracao: {ausentes}")

    for papel in ("model", "batch"):
        if not config[papel].get("producer_stage"):
            raise ErroConfiguracao(f"O papel '{papel}' exige a chave 'producer_stage'.")
    if config["model"]["producer_stage"] == config["batch"]["producer_stage"]:
        raise ErroConfiguracao("As etapas produtoras do modelo e do lote devem ser diferentes.")

    invalidas = [metrica for metrica in config["metrics"] if metrica not in METRICAS]
    if invalidas:
        raise ErroConfiguracao(f"Metricas invalidas: {invalidas}. Esperado: {list(METRICAS)}.")

    degradacao = config.get("degradation", {})
    if degradacao:
        if degradacao.get("metric") not in config["metrics"]:
            raise ErroConfiguracao("A metrica de degradacao deve constar em 'metrics'.")
        if not numero(degradacao.get("min_delta")) or float(degradacao["min_delta"]) <= 0:
            raise ErroConfiguracao("'min_delta' deve ser um numero positivo.")
    return config


def carregar_manifesto(caminho: Path) -> dict[str, Any]:
    """Le um manifesto de entrada e exige o produtor com sucesso."""
    if not caminho.is_file():
        raise ErroAvaliacao(f"Manifesto de entrada nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        manifesto = json.load(arquivo)

    if manifesto.get("status") != "success":
        raise ErroAvaliacao(f"Produtor {manifesto.get('run_id')} nao concluiu com sucesso.")
    if versao_maior(manifesto.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroAvaliacao("Versao principal do contrato do manifesto incompativel.")
    return manifesto


def separar_manifestos(
    manifestos: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identifica o manifesto do modelo e o do lote pela etapa produtora."""
    etapa_modelo = config["model"]["producer_stage"]
    etapa_lote = config["batch"]["producer_stage"]
    esperadas = (etapa_modelo, etapa_lote)

    encontrados: dict[str, dict[str, Any]] = {}
    for manifesto in manifestos:
        etapa = manifesto.get("stage")
        if etapa in esperadas:
            if etapa in encontrados:
                raise ErroAvaliacao(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroAvaliacao(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_modelo], encontrados[etapa_lote]


def artefato(manifesto: dict[str, Any], tipo: str) -> dict[str, Any]:
    """Devolve o descritor de um artefato de saida pelo tipo."""
    for item in manifesto.get("outputs", []):
        if item.get("artifact_type") == tipo:
            return item
    raise ErroAvaliacao(f"Artefato '{tipo}' ausente no manifesto {manifesto.get('run_id')}.")


def verificar_artefato(item: dict[str, Any]) -> Path:
    """Confere existencia e integridade de um artefato antes de consumi-lo."""
    caminho = Path(item.get("path", ""))
    if not caminho.is_file():
        raise ErroAvaliacao(f"Artefato nao encontrado: {caminho}")
    esperado = item.get("sha256")
    if esperado and sha256_arquivo(caminho) != esperado:
        raise ErroAvaliacao(f"Hash divergente para o artefato: {caminho}")
    return caminho


def verificar_compatibilidade(
    config: dict[str, Any], manifesto_modelo: dict[str, Any], manifesto_lote: dict[str, Any]
) -> None:
    """Confere que modelo e lote pertencem ao mesmo dataset da configuracao."""
    dataset_id = config["dataset_id"]
    for manifesto in (manifesto_modelo, manifesto_lote):
        if manifesto.get("dataset_id") != dataset_id:
            raise ErroAvaliacao(
                f"dataset_id divergente no manifesto {manifesto.get('run_id')}: "
                f"{manifesto.get('dataset_id')} != {dataset_id}"
            )


def registrar_conversores_do_treino() -> None:
    """Disponibiliza os conversores do treinamento no namespace corrente."""
    namespace = sys.modules.get("__main__")
    if namespace is None:
        return
    raiz_src = str(Path(__file__).resolve().parents[1])
    if raiz_src not in sys.path:
        sys.path.insert(0, raiz_src)

    from monitoramento import treinamento

    for nome in CONVERSORES_DO_TREINO:
        if not hasattr(namespace, nome):
            setattr(namespace, nome, getattr(treinamento, nome))


def carregar_pipeline(caminho: Path) -> Any:
    """Carrega o pipeline persistido, ajustando o namespace se necessario."""
    try:
        return joblib.load(caminho)
    except (AttributeError, ImportError):
        registrar_conversores_do_treino()
        return joblib.load(caminho)


def ler_lote(caminho: Path) -> pd.DataFrame:
    """Le o lote como texto tratando campo vazio como ausencia, como no treino."""
    base = pd.read_csv(caminho, dtype=str, na_filter=False)
    return base.replace("", np.nan)


def prever(
    pipeline: Any, metadata: dict[str, Any], lote: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray | None]:
    """Aplica o pipeline do treino e devolve a predicao e o score."""
    colunas = list(metadata.get("features", []))
    if not colunas:
        raise ErroAvaliacao("Metadados do modelo sem lista de features.")
    if IDENTIFICADOR not in lote.columns:
        raise ErroAvaliacao(f"Identificador ausente no lote: {IDENTIFICADOR}")

    alvo = metadata.get("target", {}).get("column")
    if alvo and alvo in lote.columns:
        raise ErroAvaliacao(f"Lote de producao nao deve conter o alvo: {alvo}")

    ausentes = [coluna for coluna in colunas if coluna not in lote.columns]
    if ausentes:
        raise ErroAvaliacao(f"Features ausentes no lote: {ausentes}")

    entrada = lote[colunas]

    if hasattr(pipeline, "predict_proba"):
        positivo = metadata.get("target", {}).get("positive_value", 1)
        classes = list(getattr(pipeline, "classes_", []))
        indice = classes.index(positivo) if positivo in classes else 1
        score = np.asarray(pipeline.predict_proba(entrada))[:, indice]
        limiar = float(metadata.get("prediction_threshold", 0.5))
        predito = (score >= limiar).astype(int)
        return predito, score

    predito = np.asarray(pipeline.predict(entrada)).astype(int)
    return predito, None


def juntar_rotulos(identificadores: pd.Series, caminho: Path, alvo: str) -> np.ndarray:
    """Alinha os rotulos ao lote pelo identificador, preservando a ordem."""
    if not identificadores.is_unique:
        raise ErroAvaliacao("O lote possui identificadores repetidos.")
    rotulos = pd.read_csv(caminho, dtype=str, na_filter=False)
    if IDENTIFICADOR not in rotulos.columns:
        raise ErroAvaliacao(f"Identificador ausente nos rotulos: {IDENTIFICADOR}")
    if alvo not in rotulos.columns:
        raise ErroAvaliacao(f"Coluna alvo ausente nos rotulos: {alvo}")

    numericos = pd.to_numeric(rotulos[alvo], errors="coerce")
    if numericos.isna().any():
        raise ErroAvaliacao("Os rotulos contem valores ausentes ou nao numericos.")

    mapa = pd.Series(numericos.to_numpy(), index=rotulos[IDENTIFICADOR].to_numpy())
    alinhado = identificadores.map(mapa)
    if alinhado.isna().any():
        raise ErroAvaliacao("Rotulos ausentes para alguns registros do lote.")
    return alinhado.to_numpy().astype(int)


def calcular_metricas(
    alvo: np.ndarray,
    predito: np.ndarray,
    score: np.ndarray | None,
    metricas: tuple[str, ...],
) -> dict[str, float | None]:
    """Calcula as metricas de classificacao de um conjunto de rotulos."""
    resultado: dict[str, float | None] = {}
    for metrica in metricas:
        if metrica == "roc_auc":
            if score is not None and len(np.unique(alvo)) > 1:
                resultado["roc_auc"] = round(float(roc_auc_score(alvo, score)), 6)
            else:
                resultado["roc_auc"] = None
        elif metrica == "accuracy":
            resultado["accuracy"] = round(float(accuracy_score(alvo, predito)), 6)
        elif metrica == "precision":
            resultado["precision"] = round(
                float(precision_score(alvo, predito, zero_division=0)), 6
            )
        elif metrica == "recall":
            resultado["recall"] = round(float(recall_score(alvo, predito, zero_division=0)), 6)
        elif metrica == "f1":
            resultado["f1"] = round(float(f1_score(alvo, predito, zero_division=0)), 6)
    return resultado


def comparar(
    referencia: dict[str, float | None],
    alterado: dict[str, float | None],
    metrica: str,
    min_delta: float,
) -> dict[str, Any]:
    """Compara os conceitos e aplica o criterio de degradacao declarado."""
    deltas: dict[str, float | None] = {}
    for nome, valor_ref in referencia.items():
        valor_alt = alterado.get(nome)
        if valor_ref is None or valor_alt is None:
            deltas[nome] = None
        else:
            deltas[nome] = round(valor_alt - valor_ref, 6)

    variacao = deltas.get(metrica)
    detectada = variacao is not None and variacao <= -float(min_delta)
    return {
        "metric": metrica,
        "min_delta": float(min_delta),
        "deltas": deltas,
        "metric_delta": variacao,
        "degradation_detected": bool(detectada),
    }


def montar_conclusoes(
    referencia: dict[str, float | None],
    alterado: dict[str, float | None],
    comparacao: dict[str, Any],
    conceito: dict[str, Any] | None,
) -> list[str]:
    """Registra as conclusoes tecnicas da avaliacao, sem inferir causalidade."""
    conclusoes: list[str] = [
        "O contraste usa os mesmos registros e as mesmas features: apenas a relacao entre "
        "perfil e desfecho difere entre o conceito original e o cenario de Concept Drift, "
        "isolando a mudanca de conceito (P(y|X)) da mudanca de distribuicao (P(X))."
    ]

    auc_ref = referencia.get("roc_auc")
    auc_alt = alterado.get("roc_auc")
    if auc_ref is not None and auc_alt is not None:
        conclusoes.append(
            f"No conceito original o modelo tem ROC AUC {auc_ref:.4f}; no cenario de "
            f"Concept Drift, {auc_alt:.4f} (variacao {auc_alt - auc_ref:+.4f})."
        )

    metrica = comparacao["metric"]
    variacao = comparacao["metric_delta"]
    if variacao is None:
        conclusoes.append(
            f"A metrica de comparacao '{metrica}' nao pode ser calculada para os dois conceitos."
        )
    elif comparacao["degradation_detected"]:
        conclusoes.append(
            f"A variacao de {metrica} ({variacao:+.4f}) atinge o criterio declarado de queda "
            f"(<= -{comparacao['min_delta']}), evidenciando degradacao de desempenho associada "
            "a mudanca de conceito."
        )
    else:
        conclusoes.append(
            f"A variacao de {metrica} ({variacao:+.4f}) nao atinge o criterio de degradacao "
            f"declarado (<= -{comparacao['min_delta']})."
        )

    conclusoes.append(
        "Data Drift e Concept Drift sao distintos: o Data Drift muda a distribuicao das "
        "entradas (P(X)) e e detectado sem rotulos; o Concept Drift muda a relacao entre "
        "entradas e alvo (P(y|X)) e so pode ser medido com rotulos compativeis, como neste "
        "contraste. A queda de desempenho e uma consequencia possivel do Concept Drift, nao "
        "a sua definicao."
    )

    if conceito and conceito.get("description"):
        conclusoes.append(f"Mudanca simulada: {conceito['description']}")
    return conclusoes


def montar_medidas(
    comparacao: dict[str, Any],
    referencia: dict[str, float | None],
    alterado: dict[str, float | None],
) -> list[dict[str, Any]]:
    """Converte a comparacao em medidas do contrato de metricas."""
    metrica = comparacao["metric"]
    valor_ref = referencia.get(metrica)
    valor_alt = alterado.get(metrica)
    disponibilidade = (
        "available" if valor_ref is not None and valor_alt is not None else "not_available"
    )
    return [
        {
            "name": f"{metrica}_referencia",
            "value": valor_ref,
            "unit": "metrica",
            "availability": disponibilidade,
        },
        {
            "name": f"{metrica}_concept_drift",
            "value": valor_alt,
            "unit": "metrica",
            "availability": disponibilidade,
        },
        {
            "name": f"variacao_{metrica}",
            "value": comparacao["metric_delta"],
            "unit": "variacao",
            "availability": disponibilidade,
        },
        {
            "name": "degradacao_detectada",
            "value": 1 if comparacao["degradation_detected"] else 0,
            "unit": "indicador",
            "availability": "available",
        },
    ]


def montar_documento(
    *,
    run_id: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
    modelo_id: str | None,
    referencia_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
    caminho_ref: Path,
    caminho_alt: Path,
    linhas: int,
    referencia: dict[str, float | None],
    alterado: dict[str, float | None],
    comparacao: dict[str, Any],
    conceito: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reune o documento da avaliacao com metricas, comparacao e interpretacao."""
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "dataset_id": config["dataset_id"],
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "tool": {"name": "scikit-learn", "version": sklearn.__version__},
        "metrics": list(config["metrics"]),
        "positive_class": metadata.get("target", {}).get("positive_class"),
        "prediction_threshold": float(metadata.get("prediction_threshold", 0.5)),
        "contrast": {
            "reference": {
                "concept": "conceito original (rotulos de referencia)",
                "labels_path": caminho_ref.as_posix(),
                "row_count": int(linhas),
                "metrics": referencia,
            },
            "concept_drift": {
                "concept": "cenario de Concept Drift",
                "labels_path": caminho_alt.as_posix(),
                "row_count": int(linhas),
                "metrics": alterado,
            },
        },
        "comparison": comparacao,
        "concept_change": conceito,
        "interpretation": {
            "conclusions": montar_conclusoes(referencia, alterado, comparacao, conceito)
        },
    }


def descritor_artefato(tipo: str, caminho: Path, run_id: str) -> dict[str, Any]:
    """Descreve um artefato de saida nao tabular conforme o manifesto comum."""
    return {
        "artifact_type": tipo,
        "path": caminho.as_posix(),
        "sha256": sha256_arquivo(caminho),
        "producer_run_id": run_id,
    }


def registrar_evento(
    eventos: list[dict[str, Any]],
    *,
    run_id: str,
    dataset_id: str | None,
    model_id: str | None,
    reference_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
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
            "model_id": model_id,
            "reference_id": reference_id,
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
    manifesto_lote: dict[str, Any],
    modelo_id: str | None,
    referencia_id: str | None,
    scenario_id: str | None,
    batch_id: str | None,
    entradas: list[dict[str, Any]],
    saidas: list[dict[str, Any]],
    comparacao: dict[str, Any],
    linhas: int,
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor da avaliacao."""
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": STAGE,
        "status": "success",
        "started_at": iniciado,
        "finished_at": finalizado,
        "config_path": config_path.as_posix(),
        "inputs": entradas,
        "outputs": saidas,
        "dataset_id": config["dataset_id"],
        "schema_version": manifesto_lote.get("schema_version"),
        "model_id": modelo_id,
        "reference_id": referencia_id,
        "scenario_id": scenario_id,
        "batch_id": batch_id,
        "tool": {"name": "scikit-learn", "version": sklearn.__version__},
        "comparison": comparacao,
        "row_count": int(linhas),
        "error_code": None,
    }


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any]],
    run_id: str,
    output_dir: Path,
) -> int:
    """Executa a avaliacao e publica manifesto, eventos, metricas e desempenho."""
    output_dir.mkdir(parents=True, exist_ok=True)
    iniciado = agora_iso()
    eventos: list[dict[str, Any]] = []
    registrar_evento(
        eventos,
        run_id=run_id,
        dataset_id=config.get("dataset_id"),
        model_id=None,
        reference_id=None,
        scenario_id=None,
        batch_id=None,
        evento="started",
        nivel="INFO",
    )
    try:
        manifesto_modelo, manifesto_lote = separar_manifestos(manifestos, config)
        verificar_compatibilidade(config, manifesto_modelo, manifesto_lote)

        modelo_id = manifesto_modelo.get("model_id")
        referencia_id = manifesto_modelo.get("reference_id")
        scenario_id = manifesto_lote.get("scenario_id")
        batch_id = manifesto_lote.get("batch_id")

        item_pipeline = artefato(manifesto_modelo, config["model"]["pipeline_artifact"])
        item_metadata = artefato(manifesto_modelo, config["model"]["metadata_artifact"])
        item_features = artefato(manifesto_lote, config["batch"]["features_artifact"])
        item_labels = artefato(manifesto_lote, config["batch"]["labels_artifact"])
        item_ref_labels = artefato(manifesto_lote, config["batch"]["reference_labels_artifact"])

        caminho_pipeline = verificar_artefato(item_pipeline)
        caminho_metadata = verificar_artefato(item_metadata)
        caminho_features = verificar_artefato(item_features)
        caminho_labels = verificar_artefato(item_labels)
        caminho_ref_labels = verificar_artefato(item_ref_labels)

        with caminho_metadata.open(encoding="utf-8") as arquivo:
            metadata = json.load(arquivo)

        alvo = metadata.get("target", {}).get("column")
        if not alvo:
            raise ErroAvaliacao("Metadados do modelo sem coluna alvo.")

        pipeline = carregar_pipeline(caminho_pipeline)
        lote = ler_lote(caminho_features)
        predito, score = prever(pipeline, metadata, lote)

        rotulos_ref = juntar_rotulos(lote[IDENTIFICADOR], caminho_ref_labels, alvo)
        rotulos_alt = juntar_rotulos(lote[IDENTIFICADOR], caminho_labels, alvo)

        metricas = tuple(config["metrics"])
        valores_ref = calcular_metricas(rotulos_ref, predito, score, metricas)
        valores_alt = calcular_metricas(rotulos_alt, predito, score, metricas)

        degradacao = config.get("degradation", {})
        metrica = degradacao.get("metric", "roc_auc")
        min_delta = float(degradacao.get("min_delta", 0.02))
        comparacao = comparar(valores_ref, valores_alt, metrica, min_delta)

        conceito = manifesto_lote.get("concept_change")
        documento = montar_documento(
            run_id=run_id,
            config=config,
            metadata=metadata,
            modelo_id=modelo_id,
            referencia_id=referencia_id,
            scenario_id=scenario_id,
            batch_id=batch_id,
            caminho_ref=caminho_ref_labels,
            caminho_alt=caminho_labels,
            linhas=len(lote),
            referencia=valores_ref,
            alterado=valores_alt,
            comparacao=comparacao,
            conceito=conceito,
        )
        caminho_json = output_dir / "performance.json"
        escrever_json(caminho_json, documento)

        entradas = [
            {**item_pipeline, "run_id_origem": manifesto_modelo.get("run_id")},
            {**item_metadata, "run_id_origem": manifesto_modelo.get("run_id")},
            {**item_features, "run_id_origem": manifesto_lote.get("run_id")},
            {**item_labels, "run_id_origem": manifesto_lote.get("run_id")},
            {**item_ref_labels, "run_id_origem": manifesto_lote.get("run_id")},
        ]
        saidas = [descritor_artefato("performance_json", caminho_json, run_id)]

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {
                "run_id": run_id,
                "measurements": montar_medidas(comparacao, valores_ref, valores_alt),
            },
        )
        escrever_json(
            output_dir / "manifest.json",
            montar_manifesto(
                run_id=run_id,
                iniciado=iniciado,
                finalizado=agora_iso(),
                config_path=config_path,
                config=config,
                manifesto_lote=manifesto_lote,
                modelo_id=modelo_id,
                referencia_id=referencia_id,
                scenario_id=scenario_id,
                batch_id=batch_id,
                entradas=entradas,
                saidas=saidas,
                comparacao=comparacao,
                linhas=len(lote),
            ),
        )
        if comparacao["degradation_detected"]:
            registrar_evento(
                eventos,
                run_id=run_id,
                dataset_id=config.get("dataset_id"),
                model_id=modelo_id,
                reference_id=referencia_id,
                scenario_id=scenario_id,
                batch_id=batch_id,
                evento="drift_alert",
                nivel="WARNING",
            )
        registrar_evento(
            eventos,
            run_id=run_id,
            dataset_id=config.get("dataset_id"),
            model_id=modelo_id,
            reference_id=referencia_id,
            scenario_id=scenario_id,
            batch_id=batch_id,
            evento="completed",
            nivel="INFO",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
    except (ErroAvaliacao, ValueError, KeyError) as erro:
        registrar_evento(
            eventos,
            run_id=run_id,
            dataset_id=config.get("dataset_id"),
            model_id=None,
            reference_id=None,
            scenario_id=None,
            batch_id=None,
            evento="failed",
            nivel="ERROR",
            error_code="AVALIACAO_INVALIDA",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
        print(f"Falha na avaliacao: {erro}")
        return CODIGO_ERRO

    metrica = comparacao["metric"]
    print(
        f"Avaliacao de Concept Drift: {metrica} "
        f"{valores_ref.get(metrica)} (referencia) -> {valores_alt.get(metrica)} "
        f"({comparacao['metric_delta']:+})"
    )
    print(f"  cenario {scenario_id} / lote {batch_id} (modelo {modelo_id})")
    print(f"  degradacao detectada: {comparacao['degradation_detected']}")
    print(f"  documento -> {caminho_json.as_posix()}")
    return CODIGO_SUCESSO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara o desempenho do modelo entre o conceito original e o Concept Drift."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/avaliacao_concept.json"),
        help="JSON da etapa.",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        action="append",
        default=None,
        help="Manifesto de entrada (treinamento e simulacao). Repetivel.",
    )
    parser.add_argument("--run-id", default=None, help="Identificador unico da execucao.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio novo da execucao.")
    args = parser.parse_args()

    try:
        config = carregar_config(args.config)
    except ErroConfiguracao as erro:
        print(f"Falha na avaliacao: {erro}")
        sys.exit(CODIGO_ERRO)

    entradas = args.input_manifest or []
    if len(entradas) < 2:
        print("Informe dois manifestos via --input-manifest (um do treinamento e um da simulacao).")
        sys.exit(CODIGO_ERRO)

    try:
        manifestos = [carregar_manifesto(caminho) for caminho in entradas]
    except ErroAvaliacao as erro:
        print(f"Falha na avaliacao: {erro}")
        sys.exit(CODIGO_ERRO)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id
    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(processar(args.config, config, manifestos, run_id, output_dir))


if __name__ == "__main__":
    main()
