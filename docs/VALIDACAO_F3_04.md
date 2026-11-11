# Validação do dashboard — F3-04 / issue #18

## Escopo

Revalidação das duas correções de resiliência do PR #42, sobre a stack
F3-01 → F3-02 → F3-03 → F3-04. O PR #43 recebe a mesma correção do dashboard,
preservando a documentação de governança. A conclusão técnica abaixo se refere
à implementação na stack; a issue permanece aberta até a integração dos PRs.

## Critérios de aceite

| Critério da issue #18 | Resultado e evidência |
|---|---|
| Ambiente de monitoramento funcional | Atendido. AppTest executa `main()` com as cinco abas, sem exceções, tanto com fontes válidas quanto com eventos corrompidos e alertas indisponíveis. |
| Saúde do pipeline visualizável | Atendido. A interface exibe execuções, taxa de sucesso, falhas e histórico; eventos não objetos não impedem outras execuções. |
| Métricas operacionais/infraestrutura visíveis | Atendido. Teste integrado confere ROC AUC, medidas por execução, espaço livre e capacidade de escrita nos resultados persistidos de F3-02. |
| Drift e alertas demonstrados | Atendido. Logs, saúde e alertas são gerados pelos componentes reais sobre entradas sintéticas; a interface exibe drift, execução de origem e alertas críticos. Relatório ausente ou inválido exibe `—`, enquanto zero confirmado permanece `0`. |
| Evidências visuais e instruções de acesso registradas | Atendido. Captura histórica no comentário do PR e comando de acesso abaixo; testes atuais de interface verificam o comportamento corrigido nas cinco abas. |

A [captura já registrada no PR #42](https://github.com/crisleymarques/postech-ml-challenge-fase-4/pull/42#issuecomment-6776070552)
é anterior a estas correções; não é uma nova captura do código corrigido.
Os resultados de F3-03 são produzidos e consumidos nos testes, atendendo à
dependência funcional exigida pela issue #18 dentro da stack.

## Acesso

Na raiz do projeto, com o ambiente instalado:

```sh
uv run streamlit run src/monitoramento/dashboard.py
```

Abrir o endereço local informado pelo Streamlit (por padrão,
`http://localhost:8501`). Os caminhos de execuções, logs, saúde e alertas podem
ser ajustados na barra lateral. O painel é somente leitura.

## Regressões e reprodução

```sh
uv run pytest tests/test_dashboard.py tests/test_dashboard_resiliencia.py tests/test_alertas_dashboard_regressoes.py tests/test_saude_dashboard_regressoes.py tests/test_concept_observabilidade.py tests/test_publicacao_consumidores.py -q
uv run pytest -q
uv run ruff check .
```

Em `test_dashboard_resiliencia.py`, 13 casos reproduziram falhas antes da correção.
A suíte nova contém 15 casos, incluindo leitura com erro de acesso/codificação,
eventos inválidos entre eventos válidos, indisponibilidade versus zero confirmado
e renderização integrada de saúde, infraestrutura, drift e alertas.

Resultado da suíte completa no PR #42 corrigido: **632 testes aprovados**.
Ruff, formatação dos arquivos Python alterados e `git diff --check` aprovados.

A validação usa Python 3.12 e dependências já disponíveis no ambiente de revisão,
incluindo Streamlit 1.66.1. Não foi feita nova sincronização pelo `uv.lock`.
Os dados são sintéticos; este registro não afirma reexecução do dataset completo
nem substitui as demais entregas do desafio, como governança e vídeo STAR.
