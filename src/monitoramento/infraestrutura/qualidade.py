"""Barreira de qualidade e identidade compartilhada pelos consumidores do modelo."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def verificar_metadata(
    metadata: dict[str, Any], modelo: dict[str, Any], lote: dict[str, Any], *, erro: type[Exception]
) -> None:
    """Exige identidade e schema consistentes antes de carregar o modelo."""
    schema = modelo.get("schema_version")
    if (
        not isinstance(schema, str)
        or not schema.strip()
        or any(origem.get("schema_version") != schema for origem in (metadata, lote))
    ):
        raise erro("schema_version ausente ou divergente entre modelo, metadados e lote.")
    for campo in ("dataset_id", "model_id", "reference_id"):
        valor = modelo.get(campo)
        if not isinstance(valor, str) or not valor.strip() or metadata.get(campo) != valor:
            raise erro(f"{campo} ausente ou divergente nos metadados do modelo.")


def verificar_aprovacao(
    validacao: dict[str, Any],
    lote: dict[str, Any],
    features: dict[str, Any],
    *,
    erro: type[Exception],
) -> None:
    """Vincula a aprovacao de producao ao lote e ao CSV exato consumido."""
    qualidade = validacao.get("quality")
    if not validacao.get("run_id") or not isinstance(qualidade, dict):
        raise erro("Manifesto de validacao sem aprovacao de qualidade.")
    if qualidade.get("approved") is not True:
        raise erro("Qualidade do lote nao aprovada.")
    for campo in ("dataset_id", "schema_version", "scenario_id", "batch_id"):
        valor = lote.get(campo)
        if not isinstance(valor, str) or not valor.strip() or validacao.get(campo) != valor:
            raise erro(f"{campo} divergente na aprovacao do lote.")
        if campo in ("dataset_id", "schema_version") and qualidade.get(campo) != valor:
            raise erro(f"{campo} divergente no resumo de qualidade.")
    entradas = [i for i in validacao.get("inputs", []) if i.get("artifact_type") == "features"]
    aprovados = [i for i in qualidade.get("artifacts", []) if i.get("artifact_type") == "features"]
    if len(entradas) != 1 or len(aprovados) != 1:
        raise erro("Aprovacao deve identificar um unico artefato features.")
    entrada, aprovado = entradas[0], aprovados[0]
    if (
        aprovado.get("approved") is not True
        or aprovado.get("sha256") != features.get("sha256")
        or entrada.get("sha256") != features.get("sha256")
        or not isinstance(entrada.get("path"), str)
        or Path(entrada["path"]).resolve() != Path(features["path"]).resolve()
        or not lote.get("run_id")
        or entrada.get("producer_run_id") != lote["run_id"]
        or features.get("producer_run_id") != lote["run_id"]
    ):
        raise erro("Aprovacao nao corresponde ao hash, caminho ou produtor das features.")
