"""Verifica a entrada do pacote de alertas pela linha de comando."""

import subprocess
import sys


def test_cli_exibe_ajuda() -> None:
    resultado = subprocess.run(
        [sys.executable, "-m", "monitoramento.observabilidade.alertas", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resultado.returncode == 0
    assert "--saude" in resultado.stdout
    assert "--logs-resumo" in resultado.stdout
