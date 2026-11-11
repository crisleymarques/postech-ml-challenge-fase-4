"""Caminhos padrao e constantes do painel de monitoramento."""

from pathlib import Path

RUNS_DIR_PADRAO = Path("reports/runs")
LOGS_RESUMO_PADRAO = Path("reports/logs/resumo.json")
SAUDE_PADRAO = Path("reports/saude/saude.json")
ALERTAS_PADRAO = Path("reports/alertas/alertas.json")
STATUS_SUCESSO = "success"
STATUS_REJEICAO = "rejected"
STATUS_FALHA = "failed"
STATUS_INCOMPLETO = "incomplete"
STATUS_TERMINAL = {
    "completed": STATUS_SUCESSO,
    "rejected": STATUS_REJEICAO,
    "failed": STATUS_FALHA,
}
SITUACAO_ATENCAO = "atencao"
SITUACAO_CRITICO = "critico"
SEVERIDADE_WARNING = "warning"
SEVERIDADE_CRITICAL = "critical"
CATEGORIA_EXECUCAO = "execucao"
CATEGORIA_AGREGADO = "agregado"
NAO_DISPONIVEL = "indisponivel"
