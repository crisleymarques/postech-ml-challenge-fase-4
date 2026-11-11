"""Conversoes e rotulos compartilhados pela apresentacao."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..config import NAO_DISPONIVEL


def _tabela(linhas: list[dict[str, Any]]) -> pd.DataFrame:
    """Converte uma lista de registros em DataFrame, tolerando lista vazia."""
    return pd.DataFrame(linhas) if linhas else pd.DataFrame()


def _origem_artefato(artefato: dict[str, Any] | None, disponivel: bool) -> str:
    """Descreve a origem e o horario de atualizacao de um artefato."""
    if not disponivel or artefato is None:
        return NAO_DISPONIVEL
    return str(artefato.get("generated_at") or "sem horario")
