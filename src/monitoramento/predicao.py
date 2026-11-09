"""Predicao sobre o lote de producao simulado (issue F2-02).

Le o manifesto do treinamento e o manifesto da simulacao, confere a versao de
contrato, o `dataset_id` e o hash dos artefatos, e aplica o mesmo pipeline
persistido no treino sobre as features do lote. A leitura do lote repete o
tratamento do treinamento (texto com campo vazio tratado como ausencia), de
modo que o pre-processamento seja consistente. Publica `predictions.csv` com
`record_id`, `prediction` e `score`, alem do manifesto do produtor
(`stage: predicao`) com o modelo, a referencia, o cenario, o lote e o
significado do score.

Codigos de saida: 0 sucesso, 1 erro tecnico ou incompatibilidade de artefatos.

Uso:
    uv run python src/monitoramento/predicao.py \
        --config configs/predicao.json \
        --input-manifest reports/runs/<treinamento>/manifest.json \
        --input-manifest reports/runs/<simulacao>/manifest.json
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

CONTRACT_VERSION = "1.0"
STAGE = "predicao"
IDENTIFICADOR = "record_id"

CODIGO_SUCESSO = 0
CODIGO_ERRO = 1

COLUNAS_SAIDA = (IDENTIFICADOR, "prediction", "score")

# Conversores de pre-processamento do treinamento que podem ter sido
# serializados no namespace `__main__` quando a etapa roda como script.
CONVERSORES_DO_TREINO = ("converter_numericas", "converter_categoricas")


class ErroPredicao(Exception):
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


def carregar_config(caminho: Path) -> dict[str, Any]:
    """Le e valida minimamente a configuracao da etapa."""
    if not caminho.is_file():
        raise ErroConfiguracao(f"Configuracao nao encontrada: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        config = json.load(arquivo)

    if versao_maior(config.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroConfiguracao("Versao principal do contrato da configuracao incompativel.")

    obrigatorios = ("dataset_id", "model", "batch", "outputs")
    ausentes = [chave for chave in obrigatorios if not config.get(chave)]
    if ausentes:
        raise ErroConfiguracao(f"Chaves ausentes na configuracao: {ausentes}")

    for papel in ("model", "batch"):
        if not config[papel].get("producer_stage"):
            raise ErroConfiguracao(f"O papel '{papel}' exige a chave 'producer_stage'.")
    if config["model"]["producer_stage"] == config["batch"]["producer_stage"]:
        raise ErroConfiguracao("As etapas produtoras do modelo e do lote devem ser diferentes.")
    return config


def carregar_manifesto(caminho: Path) -> dict[str, Any]:
    """Le um manifesto de entrada e exige o produtor com sucesso."""
    if not caminho.is_file():
        raise ErroPredicao(f"Manifesto de entrada nao encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as arquivo:
        manifesto = json.load(arquivo)

    if manifesto.get("status") != "success":
        raise ErroPredicao(f"Produtor {manifesto.get('run_id')} nao concluiu com sucesso.")
    if versao_maior(manifesto.get("contract_version")) != versao_maior(CONTRACT_VERSION):
        raise ErroPredicao("Versao principal do contrato do manifesto incompativel.")
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
                raise ErroPredicao(f"Manifesto duplicado para a etapa {etapa}.")
            encontrados[etapa] = manifesto

    ausentes = [etapa for etapa in esperadas if etapa not in encontrados]
    if ausentes:
        raise ErroPredicao(f"Manifestos ausentes para as etapas: {ausentes}")
    return encontrados[etapa_modelo], encontrados[etapa_lote]


def artefato(manifesto: dict[str, Any], tipo: str) -> dict[str, Any]:
    """Devolve o descritor de um artefato de saida pelo tipo."""
    for item in manifesto.get("outputs", []):
        if item.get("artifact_type") == tipo:
            return item
    raise ErroPredicao(f"Artefato '{tipo}' ausente no manifesto {manifesto.get('run_id')}.")


def verificar_artefato(item: dict[str, Any]) -> Path:
    """Confere existencia e integridade de um artefato antes de consumi-lo."""
    caminho = Path(item.get("path", ""))
    if not caminho.is_file():
        raise ErroPredicao(f"Artefato nao encontrado: {caminho}")
    esperado = item.get("sha256")
    if esperado and sha256_arquivo(caminho) != esperado:
        raise ErroPredicao(f"Hash divergente para o artefato: {caminho}")
    return caminho


def verificar_compatibilidade(
    config: dict[str, Any], manifesto_modelo: dict[str, Any], manifesto_lote: dict[str, Any]
) -> None:
    """Confere que modelo e lote pertencem ao mesmo dataset da configuracao."""
    dataset_id = config["dataset_id"]
    for manifesto in (manifesto_modelo, manifesto_lote):
        if manifesto.get("dataset_id") != dataset_id:
            raise ErroPredicao(
                f"dataset_id divergente no manifesto {manifesto.get('run_id')}: "
                f"{manifesto.get('dataset_id')} != {dataset_id}"
            )


def ler_lote(caminho: Path) -> pd.DataFrame:
    """Le o lote como texto tratando campo vazio como ausencia, como no treino."""
    base = pd.read_csv(caminho, dtype=str, na_filter=False)
    return base.replace("", np.nan)


def registrar_conversores_do_treino() -> None:
    """Disponibiliza os conversores do treinamento no namespace corrente.

    Quando o treinamento roda como script, o pipeline serializa os conversores
    de pre-processamento pelo modulo `__main__`. Registrando esses nomes a
    partir do modulo de treinamento, o artefato ja persistido pode ser lido em
    outro processo sem qualquer alteracao no produtor.
    """
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


def prever(
    pipeline: Any, metadata: dict[str, Any], lote: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    """Aplica o pipeline do treino e monta a saida de predicoes.

    Devolve o quadro de predicoes e se o score esta disponivel.
    """
    colunas = list(metadata.get("features", []))
    if not colunas:
        raise ErroPredicao("Metadados do modelo sem lista de features.")
    if IDENTIFICADOR not in lote.columns:
        raise ErroPredicao(f"Identificador ausente no lote: {IDENTIFICADOR}")

    alvo = metadata.get("target", {}).get("column")
    if alvo and alvo in lote.columns:
        raise ErroPredicao(f"Lote de producao nao deve conter o alvo: {alvo}")

    ausentes = [coluna for coluna in colunas if coluna not in lote.columns]
    if ausentes:
        raise ErroPredicao(f"Features ausentes no lote: {ausentes}")

    entrada = lote[colunas]
    rotulos = pipeline.predict(entrada)

    if hasattr(pipeline, "predict_proba"):
        positivo = metadata.get("target", {}).get("positive_value", 1)
        classes = list(getattr(pipeline, "classes_", []))
        indice = classes.index(positivo) if positivo in classes else 1
        score = np.asarray(pipeline.predict_proba(entrada))[:, indice]
        score_disponivel = True
    else:
        score = np.full(len(entrada), np.nan)
        score_disponivel = False

    predicoes = pd.DataFrame(
        {
            IDENTIFICADOR: lote[IDENTIFICADOR].to_numpy(),
            "prediction": np.asarray(rotulos).astype(int),
            "score": np.round(score, 6),
        }
    )
    if not predicoes[IDENTIFICADOR].is_unique:
        raise ErroPredicao("O lote possui identificadores repetidos.")
    return predicoes, score_disponivel


def escrever_predicoes(caminho: Path, base: pd.DataFrame) -> None:
    """Grava as predicoes em CSV."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(caminho, index=False, encoding="utf-8")


def descritor_artefato(tipo: str, caminho: Path, run_id: str, base: pd.DataFrame) -> dict[str, Any]:
    """Descreve um artefato tabular conforme o manifesto comum."""
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


def montar_medidas(predicoes: pd.DataFrame, score_disponivel: bool) -> list[dict[str, Any]]:
    """Converte as predicoes geradas em medidas do contrato de metricas."""
    total = int(len(predicoes))
    positivos = int(predicoes["prediction"].sum()) if total else 0
    if score_disponivel and total:
        score_medio: float | None = round(float(predicoes["score"].mean()), 6)
        disponibilidade_score = "available"
    else:
        score_medio = None
        disponibilidade_score = "not_available"
    return [
        {
            "name": "predicoes_geradas",
            "value": total,
            "unit": "linhas",
            "availability": "available",
        },
        {
            "name": "taxa_positivos",
            "value": round(positivos / total, 6) if total else None,
            "unit": "proporcao",
            "availability": "available" if total else "not_available",
        },
        {
            "name": "score_medio",
            "value": score_medio,
            "unit": "probabilidade",
            "availability": disponibilidade_score,
        },
    ]


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
    score: dict[str, Any],
    linhas: int,
) -> dict[str, Any]:
    """Reune o manifesto comum do produtor das predicoes."""
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
        "row_count": int(linhas),
        "score": score,
        "error_code": None,
    }


def processar(
    config_path: Path,
    config: dict[str, Any],
    manifestos: list[dict[str, Any]],
    run_id: str,
    output_dir: Path,
    *,
    analysis_dir: Path | None = None,
) -> int:
    """Executa a predicao e publica manifesto, eventos, metricas e predicoes."""
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

        caminho_pipeline = verificar_artefato(item_pipeline)
        caminho_metadata = verificar_artefato(item_metadata)
        caminho_features = verificar_artefato(item_features)

        with caminho_metadata.open(encoding="utf-8") as arquivo:
            metadata = json.load(arquivo)

        pipeline = carregar_pipeline(caminho_pipeline)
        lote = ler_lote(caminho_features)
        predicoes, score_disponivel = prever(pipeline, metadata, lote)

        if scenario_id is None or batch_id is None:
            raise ErroPredicao("O manifesto do lote nao informa scenario_id e batch_id.")

        destino = (
            (analysis_dir or Path(config["outputs"]["analysis_dir"]))
            / str(scenario_id)
            / str(batch_id)
            / str(modelo_id)
        )
        caminho_predicoes = destino / "predictions.csv"
        escrever_predicoes(caminho_predicoes, predicoes)

        entradas = [
            {**item_pipeline, "run_id_origem": manifesto_modelo.get("run_id")},
            {**item_metadata, "run_id_origem": manifesto_modelo.get("run_id")},
            {**item_features, "run_id_origem": manifesto_lote.get("run_id")},
        ]
        saidas = [descritor_artefato("predictions", caminho_predicoes, run_id, predicoes)]

        score = {
            "available": bool(score_disponivel),
            "column": "score",
            "meaning": metadata.get("score_meaning"),
            "positive_value": metadata.get("target", {}).get("positive_value"),
            "prediction_threshold": metadata.get("prediction_threshold"),
        }

        escrever_json(output_dir / "config.json", config)
        escrever_json(
            output_dir / "metrics.json",
            {
                "run_id": run_id,
                "measurements": montar_medidas(predicoes, score_disponivel),
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
                score=score,
                linhas=len(predicoes),
            ),
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
    except (ErroPredicao, ValueError, KeyError) as erro:
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
            error_code="PREDICAO_INVALIDA",
        )
        escrever_eventos(output_dir / "events.jsonl", eventos)
        print(f"Falha na predicao: {erro}")
        return CODIGO_ERRO

    print(f"Predicoes geradas: {len(predicoes)} linhas (modelo {modelo_id})")
    print(f"  cenario {scenario_id} / lote {batch_id}")
    print(f"  score disponivel: {score_disponivel} ({metadata.get('score_meaning')})")
    print(f"  predicoes -> {caminho_predicoes.as_posix()}")
    print(f"  manifesto -> {output_dir.as_posix()}/manifest.json")
    return CODIGO_SUCESSO


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera predicoes sobre o lote de producao.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/predicao.json"), help="JSON da etapa."
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
    parser.add_argument(
        "--analysis-dir", type=Path, default=None, help="Diretorio raiz do conjunto simulado."
    )
    args = parser.parse_args()

    try:
        config = carregar_config(args.config)
    except ErroConfiguracao as erro:
        print(f"Falha na predicao: {erro}")
        sys.exit(CODIGO_ERRO)

    entradas = args.input_manifest or []
    if len(entradas) < 2:
        print("Informe dois manifestos via --input-manifest (um do treinamento e um da simulacao).")
        sys.exit(CODIGO_ERRO)

    try:
        manifestos = [carregar_manifesto(caminho) for caminho in entradas]
    except ErroPredicao as erro:
        print(f"Falha na predicao: {erro}")
        sys.exit(CODIGO_ERRO)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or Path("reports/runs") / run_id
    if output_dir.exists():
        print(f"Diretorio de execucao ja existe: {output_dir}")
        sys.exit(CODIGO_ERRO)

    sys.exit(
        processar(
            args.config,
            config,
            manifestos,
            run_id,
            output_dir,
            analysis_dir=args.analysis_dir,
        )
    )


if __name__ == "__main__":
    main()
