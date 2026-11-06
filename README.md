# postech-ml-challenge-fase-4

Projeto do Tech Challenge da Fase 4 da pós-graduação em Machine Learning Engineering da PosTech, desenvolvido em equipe de três integrantes.

Este projeto é a camada de sustentação e confiabilidade de um modelo de Credit
Scoring: contratos de qualidade de dados, simulação de degradação (Data e Concept
Drift), detecção com observabilidade e governança sob a LGPD. A documentação reside
em [docs/](docs/), com a [arquitetura e os contratos entre componentes](docs/ARQUITETURA.md)
e a [seleção e análise do dataset](docs/DATASET.md). Esses documentos são atualizados
conforme o grupo define escopo e implementação.

## Ambiente de desenvolvimento

O projeto utiliza Python 3.12 e uv para gerenciar o ambiente e as
dependências.

### Pré-requisitos

- Python 3.12.
- uv instalado.

### Reprodução do ambiente

Na raiz do repositório, execute:
```powershell
uv sync --locked
```
Esse comando sincroniza o ambiente .venv com as versões registradas
no uv.lock. Não é necessário ativar o ambiente ao usar uv run.

### Dependências

| Ferramenta | Finalidade |
|---|---|
| NumPy e pandas | Manipulação e processamento de dados |
| SciPy | Testes estatísticos |
| scikit-learn | Preparação dos dados e modelo baseline |
| Pandera | Contratos e validação de dados |
| Evidently | Detecção de drift e relatórios |
| Streamlit | Dashboard de monitoramento |
| JupyterLab e ipykernel | Notebooks e experimentação |
| pytest | Testes automatizados |
| Ruff | Análise estática e formatação de código |

A instalação dessas ferramentas prepara o ambiente; não representa
a implementação dos componentes do pipeline.

### Configuração e reprodutibilidade

- .python-version: seleciona Python 3.12 para o ambiente local.
- pyproject.toml: declara dependências e configura Ruff e pytest.
- uv.lock: registra as versões resolvidas das dependências.
- .venv/: ambiente local, excluído do versionamento.

O pandas está limitado a >=2.2,<3.1 devido à incompatibilidade de
importação observada entre Evidently e pandas 3.1. A verificação de
importações passou com pandas 3.0.6.

## Estrutura inicial
```text
src/
└── monitoramento/    # Código do projeto
tests/               # Testes automatizados
notebooks/           # Exploração e experimentação
reports/             # Relatórios e evidências
models/
data/
├── reference/       # Dados de referência
└── analysis/        # Dados de análise/produção simulada
```
As pastas inicialmente vazias contêm .gitkeep para preservar a
estrutura no Git.

O conteúdo de data/reference/ e data/analysis/ está excluído do
versionamento, exceto pelos arquivos .gitkeep. Os arquivos brutos ficam em
data/reference/raw/ e a base derivada em data/reference/sample/.

## Verificações do ambiente

Execute no PowerShell, na raiz do projeto:
```powershell
uv pip check
uv run ruff check .
uv run python -c "import numpy, pandas, scipy, sklearn, pandera, evidently, streamlit; print('Importações OK')"
uv run pytest --version
```

Durante o setup, foram verificadas a consistência das dependências,
as importações das bibliotecas e a disponibilidade de Ruff e pytest.
A análise estática também passou.

Essas verificações não validam o pipeline. Ainda não foram implementados
testes funcionais, contratos de dados, treinamento, experimentos de drift
ou dashboard.

## Arquitetura e contratos

A arquitetura definida para implementação é um pipeline local em lote:
preparação e separação → validação de dados → treinamento e avaliação;
em paralelo ao treino, a base reservada alimenta simulação → validação de
lotes → predição → detecção. Lotes inválidos são bloqueados antes de chegar
ao modelo. Todas as etapas exportam eventos e resultados para monitoramento.

O dashboard **Streamlit** foi escolhido para observabilidade, com eventos
JSONL, métricas JSON e relatórios HTML do Evidently. MLflow não integra a
arquitetura base. As métricas específicas e os limiares serão definidos pelo
grupo durante a implementação.

Os [contratos entre componentes](docs/ARQUITETURA.md) detalham o diagrama,
os formatos CSV/JSON/JSONL, manifestos, argumentos futuros dos scripts,
estados de execução e artefatos produzidos por etapa. Dados de treino,
validação e teste ficam separados da base e dos lotes de produção simulada;
a referência de drift é congelada a partir das features de treino.

Scripts, contratos Pandera, treinamento, simulação, detecção e
dashboard ainda não estão implementados. A documentação de arquitetura
não substitui as evidências funcionais nem a governança exigidas na entrega.

## Dataset — PREP-02

O dataset escolhido é o **Lending Club Loan Data**, obtido no Kaggle
(`wordsforthewise/lending-club`). O arquivo de empréstimos aceitos possui
2.260.701 linhas e 151 colunas; usamos apenas os registros com desfecho
definitivo.

Para viabilizar o trabalho, o script
[src/monitoramento/selecao_dataset.py](src/monitoramento/selecao_dataset.py)
gera de forma reprodutível uma base derivada de **100.000 linhas** em
`data/reference/sample/lending_club_sample_100k.csv`, mantendo `Fully Paid` e
`Charged Off` e preservando o eixo temporal `issue_d`.

A análise completa — origem, alvo, principais features, qualidade, candidatas a
drift e campos pessoais/sensíveis (LGPD) — está em
[docs/DATASET.md](docs/DATASET.md).

## Modelos

A pasta models/ armazena os modelos gerados localmente. Seus arquivos
não são versionados, exceto pelo .gitkeep. O procedimento de geração
será documentado quando o treinamento for implementado.

## Convenção de commits

Os commits seguem o formato:

tipo: descrição curta

| Tipo | Uso |
|---|---|
| feat | Nova funcionalidade |
| fix | Correção de erro |
| docs | Alteração na documentação |
| test | Criação ou alteração de testes |
| refactor | Reorganização de código sem mudar o comportamento |
| chore | Configuração, dependências e manutenção |

Exemplos:

- chore: configura ambiente com uv
- feat: adiciona contrato de dados
- test: verifica rejeição de lote inválido

## Estado atual

Concluídos:

- Configuração do ambiente com uv e Python 3.12.
- Instalação das dependências de execução e desenvolvimento.
- Configuração inicial de Ruff e pytest.
- Criação da estrutura de pastas e das regras de exclusão dos dados.
- Seleção e análise do dataset na PREP-02, com geração da base derivada de
  100.000 linhas e documentação em docs/DATASET.md.
- Definição da arquitetura, contratos entre etapas e estratégia
  de observabilidade com Streamlit.

Pendentes:

- Confirmação da licença de uso do dataset.
- Definição do schema de negócio, das regras rígidas de qualidade e da
  estratégia de separação das partições.
- Validação funcional do ambiente com dados sintéticos.
- Implementação do pipeline (contratos, treino, simulação, detecção, dashboard).
