# postech-ml-challenge-fase-4

Projeto do Tech Challenge da Fase 4 da pós-graduação em Machine Learning Engineering da PosTech, desenvolvido em equipe de três integrantes.

Este projeto é a camada de sustentação e confiabilidade de um modelo de Credit
Scoring: contratos de qualidade de dados, simulação de degradação (Data e Concept
Drift), detecção com observabilidade e governança sob a LGPD. A documentação reside
em [docs/](docs/), com a [arquitetura e os contratos entre componentes](docs/ARQUITETURA.md)
e a [seleção e análise do dataset](docs/DATASET.md). Esses documentos são atualizados
conforme o grupo define escopo e implementação.

## Documentação

| Documento | Conteúdo |
|---|---|
| [Arquitetura e contratos](docs/ARQUITETURA.md) | Fluxo implementado e interfaces planejadas |
| [Dataset — PREP-02](docs/DATASET.md) | Origem, download, amostragem, análise e limitações dos dados |

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
| KaggleHub | Download do arquivo bruto do Lending Club quando ausente |
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

As verificações anteriores cobriram a configuração e os testes isolados então
disponíveis. O fluxo sintético preparação → validação → treinamento está
coberto pelos testes atuais; simulação de lotes, detecção de drift e dashboard
permanecem pendentes.

## Arquitetura e contratos

A arquitetura é um pipeline local em lote: preparação e separação → validação
de dados → treinamento e avaliação, etapas implementadas; em paralelo ao treino,
a base reservada deverá alimentar simulação → validação de lotes → predição →
detecção. Lotes inválidos são bloqueados antes de chegar ao modelo. As etapas
implementadas exportam eventos e resultados para monitoramento.

O dashboard **Streamlit** foi escolhido para observabilidade, com eventos
JSONL, métricas JSON e relatórios HTML do Evidently. MLflow não integra a
arquitetura base. As métricas específicas e os limiares de drift serão definidos
antes da implementação da detecção.

Os [contratos entre componentes](docs/ARQUITETURA.md) detalham o diagrama,
os formatos CSV/JSON/JSONL, manifestos, argumentos futuros dos scripts,
estados de execução e artefatos produzidos por etapa. Dados de treino,
validação e teste ficam separados da base e dos lotes de produção simulada;
a referência de drift é congelada a partir das features de treino.

Obtenção/amostragem, preparação, validação Pandera e treinamento baseline estão
implementados. Simulação de lotes, predição, detecção de drift e dashboard
continuam planejados. As métricas históricas desta documentação não foram
reexecutadas como parte das correções do PR #32.

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

### Download e geração da amostra

Após `uv sync --locked`, execute no PowerShell, na raiz do projeto:

```powershell
uv run python src/monitoramento/selecao_dataset.py
```

Se o CSV padrão estiver ausente, o script baixa apenas
`accepted_2007_to_2018Q4.csv.gz` via KaggleHub, descompacta em diretório
temporário e confere as colunas antes de disponibilizar a origem em
`data/reference/raw/accepted_2007_to_2018Q4.csv`.
Nas próximas execuções, reutiliza esse CSV sem novo download.
A amostra é gerada novamente a cada execução, sobrescrevendo a saída escolhida.
Um `--source` personalizado inexistente gera erro, sem download automático.

Para preservar a amostra principal ao experimentar, informe outra saída:

```powershell
uv run python src/monitoramento/selecao_dataset.py --output data/reference/sample/lending_club_teste.csv
```

Caso o Kaggle exija autenticação, configure-a interativamente, sem colocar
credenciais no código:

```powershell
uv run python -c "import kagglehub; kagglehub.login()"
```

As duas execuções apresentadas pelo responsável em 07/11/2026 confirmaram
download inicial e reutilização local. Ambas reportaram 2.260.701 linhas lidas,
1.345.310 desfechos definitivos e amostra de 100.000 linhas e 30 colunas:
80.116 `Fully Paid` e 19.884 `Charged Off`.
Contagens iguais não comprovam identidade dos arquivos byte a byte.
A versão remota do dataset ainda não está fixada; um novo download futuro
pode obter uma versão diferente.

### Testes da obtenção do dataset

```powershell
uv run pytest tests/test_download_dataset.py -q
```

Para executar todos os testes disponíveis:

```powershell
uv run pytest
```

Os quatro testes de download usam simulações e conteúdo sintético, sem
acessar a rede ou o dataset real. Cobrem origem local existente, download
inicial seguido de reutilização, falha sem origem parcial e origem
personalizada ausente. O responsável informou aprovação dos quatro testes.

O `pyproject.toml` configura `--basetemp=.venv/pytest-tmp`, evitando a pasta
temporária do sistema que apresentou erro de permissão no ambiente local.
Esse diretório é exclusivo dos testes: o pytest limpa seu conteúdo a cada
execução. Essa validação não cobre o pipeline completo nem substitui contratos
de qualidade, treinamento e monitoramento.

## Preparação e separação das partições — F1-01

O componente [src/monitoramento/preparacao.py](src/monitoramento/preparacao.py)
lê a base derivada, binariza o alvo (`Charged Off` = 1, `Fully Paid` = 0) e separa
quatro partições **disjuntas** pela ordem temporal de `issue_d`, conforme os
parâmetros de [configs/preparacao.json](configs/preparacao.json) (60% treino, 20%
validação, 10% teste, 10% produção simulada):

```powershell
uv run python src/monitoramento/preparacao.py --config configs/preparacao.json
```

| Artefato | Local |
|---|---|
| Treino, validação e teste | `data/reference/<preparation_id>/{train,validation,test}.csv` |
| Base de produção simulada | `data/analysis/<preparation_id>/simulation_base.csv` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

A etapa não ajusta imputação, escala ou codificação sobre o conjunto completo:
apenas faz `trim` nos textos e mantém o alvo binarizado. Ela recusa sobrescrever
um diretório de partição já existente e encerra com código `0` (sucesso), `2`
(dados rejeitados) ou `1` (erro técnico). O contrato completo está em
[docs/ARQUITETURA.md](docs/ARQUITETURA.md) e os resultados da execução real em
[docs/DATASET.md](docs/DATASET.md).

```powershell
uv run pytest tests/test_preparacao.py -q
```

## Treinamento do modelo baseline — F1-02

O componente [src/monitoramento/treinamento.py](src/monitoramento/treinamento.py)
consome o manifesto aprovado pela validação, ajusta o pipeline no treino, seleciona pela
validação e avalia no teste, conforme os parâmetros de
[configs/treinamento.json](configs/treinamento.json):

```powershell
uv run python src/monitoramento/treinamento.py --config configs/treinamento.json --input-manifest reports/runs/<run_id_da_validacao>/manifest.json
```

| Artefato | Local |
|---|---|
| Pipeline persistido | `models/<model_id>/pipeline.pkl` |
| Metadados do modelo | `models/<model_id>/metadata.json` |
| Métricas de avaliação | `models/<model_id>/evaluation.json` |
| Referência congelada | `data/reference/<reference_id>/features.csv` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

Cada execução grava em diretórios novos e imutáveis. O `model_id` e o
`reference_id` combinam a assinatura do treino com um sufixo da execução; assim,
reexecuções e mudanças somente em validação/teste preservam as avaliações e
manifestos anteriores. Se qualquer destino já existir, a execução falha antes
de gravar modelo ou referência. A etapa aceita `--config`, `--input-manifest`,
`--run-id` e `--output-dir`, exige um manifesto de `validacao` com qualidade
aprovada, o mesmo `dataset_id` e `schema_version` configurados, e confere
novamente hashes e colunas das três partições antes do treinamento.
Preparação direta, validação rejeitada e hashes ou schemas divergentes
bloqueiam a criação do modelo. Encerra com código `0` em sucesso e `1` para
dependência rejeitada, incompatibilidade de artefatos ou erro técnico.
O código `2` é usado pela etapa de validação para identificar lote reprovado.
O contrato completo está em
[docs/ARQUITETURA.md](docs/ARQUITETURA.md).

Os conversores ficam em `monitoramento.preprocessamento`, módulo importável
necessário para carregar o pipeline em outro processo com `src` no `PYTHONPATH`.
O treinamento confere domínio binário 0/1 e rejeita ausências, infinitos e
frações antes de converter o alvo. O treinamento consome o manifesto aprovado
da validação Pandera.

```powershell
uv run pytest tests/test_treinamento.py -q
```

### Baseline: regressão logística

O baseline usa regressão logística com `class_weight="balanced"`, semente 42 e
22 features (14 numéricas e 8 categóricas) sobre as partições de referência. O
`model_id` é `baseline-logreg-820dbb61` e a referência `ref-820dbb61`. A seleção
do modelo usa `roc_auc` na validação. A classe positiva é `Charged Off` e o
limiar de decisão é `0.5`.

| Métrica | Treino | Validação | Teste |
|---|---|---|---|
| accuracy | 0.6600 | 0.6625 | 0.6626 |
| precision | 0.2965 | 0.3577 | 0.3641 |
| recall | 0.6262 | 0.5900 | 0.5658 |
| f1 | 0.4025 | 0.4454 | 0.4431 |
| roc_auc | 0.7029 | 0.6961 | 0.6891 |

As métricas são referência para comparação com modelos e cenários futuros. O
`roc_auc` próximo de 0,69 e a precisão baixa em relação ao recall indicam um
modelo simples, com muitas features sem tratamento e sem engenharia de
atributos; esses resultados serão usados como linha de base para medir a
degradação nos cenários de drift.

A pasta models/ armazena os modelos gerados localmente. Seus arquivos
não são versionados, exceto pelo .gitkeep.

## Validação do contrato de qualidade — F1-03

O componente [src/monitoramento/validacao.py](src/monitoramento/validacao.py)
consome o manifesto de um produtor (`preparacao` ou `simulacao`), confere os
hashes dos artefatos tabulares e aplica o contrato Pandera descrito em
[configs/validacao.json](configs/validacao.json):

```powershell
uv run python src/monitoramento/validacao.py --config configs/validacao.json --input-manifest reports/runs/<run_id_do_produtor>/manifest.json
```

O perfil padrão `reference` exige o alvo e valida `train`, `validation` e
`test`. Para validar um lote de produção sem alvo, use
`--profile production`; ele seleciona o artefato `features`, preserva as
colunas de identificador/cenário/lote e aplica as regras às features.

```powershell
uv run python src/monitoramento/validacao.py --config configs/validacao.json --profile production --input-manifest reports/runs/<run_id_da_simulacao>/manifest.json
```

| Artefato | Local |
|---|---|
| Relatório de qualidade | `reports/runs/<run_id>/quality.json` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

O contrato exige ao menos três regras rígidas e cobre identificador (formato),
alvo (domínio `0`/`1`), campos obrigatórios não nulos, unicidade, faixas
numéricas e domínios categóricos. Uma violação bloqueia o conjunto inteiro: o
relatório registra `approved`, `checked_rows` e a lista de `violations` com
`rule_id` e `count`, sem reescrever nem descartar registros. A etapa aceita
`--config`, `--input-manifest`, `--run-id` e `--output-dir` e encerra com código
`0` (aprovado), `2` (dados rejeitados) ou `1` (erro técnico). Valores nulos
seguem a regra de obrigatoriedade de cada campo; texto presente que não possa
ser convertido em número falha nas regras de faixa e domínio numérico. O contrato completo
está em [docs/ARQUITETURA.md](docs/ARQUITETURA.md).

A execução real sobre o manifesto da preparação aprovou 90.000 linhas
(60.000 treino, 20.000 validação e 10.000 teste), com `approved: true` e nenhuma
violação registrada.

```powershell
uv run pytest tests/test_validacao.py -q
```

## Bloqueio de ingestão de dados inválidos — F1-04

O componente
[src/monitoramento/lote_invalido.py](src/monitoramento/lote_invalido.py) gera um
lote propositalmente inválido a partir da base de simulação. Ele amostra um
conjunto determinístico (semente fixa), divide em treino, validação e teste e
injeta uma anomalia para cada tipo de regra do contrato, publicando o manifesto
do produtor (`stage: simulacao`, `status: success`) junto do lote:

```powershell
uv run python src/monitoramento/lote_invalido.py --config configs/validacao.json --source data/analysis/lending_club_100k_v1/simulation_base.csv --size 600
```

| Partição | Regra exercitada | Anomalia injetada |
|---|---|---|
| train | `R_ID_FORMATO` | identificador fora do padrão `LC#######` |
| train | `R_UNICO_record_id` | linha duplicada |
| validation | `R_NAO_NULO_loan_amnt` | `loan_amnt` vazio |
| validation | `R_FAIXA_loan_amnt` | `loan_amnt` acima do limite |
| test | `R_CATEGORIA_term` | `term` fora do domínio |
| test | `R_ALVO_DOMINIO` | `loan_status` fora de `0`/`1` |

O lote e o manifesto são gravados em
`data/analysis/<preparation_id>/lote-invalido/<batch_id>/` e
`reports/runs/<run_id>/`, ambos ignorados pelo Git. Em seguida, a validação do
lote comprova o bloqueio: o contrato reprova o conjunto inteiro e a etapa
encerra com código `2` e `error_code: QUALIDADE_REPROVADA`, registrando cada
regra violada por `rule_id`, sem aprovar nenhum artefato para consumo.

Esta demonstração usa somente o perfil `reference`, com as partições
`train`, `validation` e `test`. Configurações de produção (`features`) ou
partições não suportadas são recusadas antes de iniciar a execução; o
gerador não substitui o simulador de lotes de produção.

O campo `target_contract` de `configs/validacao.json` declara a semântica do
alvo da origem usada nesta demonstração e deve corresponder ao `target` da
preparação que gerou essa base. O gerador confere a coluna e o domínio contra
as regras de qualidade e publica esse contrato em `target` no manifesto,
preservando o vínculo exigido pela validação e pelo treinamento.

Por padrão, cada chamada gera novos identificadores de execução e lote.
Se informar `--batch-id`, use um valor novo: um destino existente é recusado
sem sobrescrever seus CSVs. O mesmo vale para `--output-dir`. Configuração
inválida termina com código `1` antes de criar o diretório de execução.
Falhas posteriores ao início, como origem ausente ou lote insuficiente,
registram eventos e manifesto terminal com `status: failed` e `error_code`,
sempre que a persistência estiver disponível. O manifesto é publicado após
os demais arquivos da execução.

```powershell
uv run python src/monitoramento/validacao.py --config configs/validacao.json --input-manifest reports/runs/<run_id_do_lote>/manifest.json
```

Resultado real da demonstração (lote de 600 linhas, 601 após a duplicata):

| rule_id | ocorrências |
|---|---|
| `R_ALVO_DOMINIO` | 1 |
| `R_CATEGORIA_term` | 1 |
| `R_FAIXA_loan_amnt` | 1 |
| `R_ID_FORMATO` | 1 |
| `R_NAO_NULO_loan_amnt` | 1 |
| `R_UNICO_record_id` | 2 |

```powershell
uv run pytest tests/test_lote_invalido.py -q
```

## Simulação temporal de Data Drift — F2-01

O componente
[src/monitoramento/simulacao/data_drift/servico.py](src/monitoramento/simulacao/data_drift/servico.py)
gera um conjunto de produção distinto da referência: parte da base reservada
para produção simulada, avança o período de emissão (`issue_d`) para
representar a passagem do tempo e aplica mudanças econômicas deterministas que
alteram a distribuição de cinco features importantes do modelo. A referência
original não é reescrita; cada execução grava um conjunto novo e publica o
manifesto do produtor (`stage: simulacao`) com o período, as variáveis
alteradas, as transformações e os parâmetros efetivos.

```powershell
uv run simular-data-drift ^
  --config configs/simulacao.json ^
  --batch-id batch-f201
```

| Variável | Operação | Parâmetro | Efeito | Média antes → depois |
|---|---|---|---|---|
| `annual_inc` | escala | 0,80 | renda cai (recessão) | 78.694,91 → 62.970,37 |
| `revol_util` | escala | 1,35 | utilização do crédito sobe | 41,97 → 56,67 |
| `dti` | deslocamento | +6,0 | comprometimento da renda sobe | 18,67 → 24,62 |
| `fico_range_low` | deslocamento | −25,0 | score cai | 704,70 → 679,69 |
| `fico_range_high` | deslocamento | −25,0 | score cai | 708,70 → 683,70 |

Cada linha recebe ainda uma variação relativa de até 5% (`jitter`), derivada
da semente fixa, para representar diferenças individuais sem comprometer a
reprodutibilidade. As features são gravadas em `features.csv` e os rótulos em
arquivo separado (`labels.csv`), conforme o contrato da etapa de simulação.

As saídas ficam em `data/analysis/<scenario_id>/<batch_id>/` e o manifesto em
`reports/runs/<run_id>/`, ambos ignorados pelo Git. Na execução real, o período
da base passou de `Jul-2017..Dec-2018` para `Jul-2019..Dec-2020` (deslocamento
de 24 meses), com 10.000 linhas e `record_id` preservado e único. Os valores
permanecem dentro das faixas do contrato de qualidade, de modo que o cenário
chega à detecção de drift sem ser reprovado por qualidade.

```powershell
uv run pytest tests/simulacao/data_drift -q
```

O deslocamento temporal deve ser um número inteiro positivo. Antes de publicar
os CSVs, o componente compara as distribuições empíricas das features com a
base de simulação e exige pelo menos duas mudanças efetivas. Parâmetros
neutros, colunas inteiramente ausentes, zeros ou limites de faixa que
neutralizem uma transformação não contam como drift. O manifesto e a métrica
`variaveis_alteradas` registram somente as features efetivamente alteradas;
identificador, alvo e coluna temporal não entram nessa contagem. Essa
conferência de alteração não substitui os testes estatísticos da detecção.

Ausências originais são preservadas, mas valores presentes não numéricos ou
não finitos nas features transformadas interrompem a simulação com
`ERRO_ENTRADA`, sem expor o conteúdo inválido nos logs nem publicar CSVs para
consumo. Assim, um valor inválido não é convertido silenciosamente em um
campo vazio que o contrato de produção poderia aceitar.

Os diretórios da execução e do lote são exclusivos: `--output-dir` e
`--batch-id` não podem reutilizar destinos existentes, inclusive em chamadas
concorrentes. Uma colisão encerra com código `1`, preservando os CSVs e os
manifestos anteriores. Use um novo identificador para repetir o cenário;
a configuração é validada antes de iniciar a execução.

Após reservar a execução, a simulação persiste `config.json` com os valores
efetivos (inclusive a seed e os caminhos sobrepostos pela CLI) e o evento
`started`. Falhas de leitura, transformação ou publicação encerram com código
`1` e registram `failed`/`ERROR`, `error_code` sanitizado, métricas vazias e
manifesto `status: failed` sem saídas aprovadas, sempre que a persistência
estiver disponível. CSVs parciais não são liberados para consumo. O manifesto
terminal é publicado por último; se a própria persistência de erro falhar,
uma mensagem controlada é emitida em `stderr`, sem anunciar sucesso.

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
- Download automático do arquivo bruto ausente e reutilização local via KaggleHub,
  com quatro testes isolados aprovados segundo o responsável e duas execuções
  reais apresentadas na conversa.
- Definição da arquitetura, contratos entre etapas e estratégia
  de observabilidade com Streamlit.
- Preparação e separação das partições de treino, validação, teste e produção
  simulada (F1-01), com manifesto, eventos e métricas, e testes em
  `tests/test_preparacao.py`.
- Treinamento e avaliação do modelo baseline de Credit Scoring (F1-02), com
  registro das métricas de classificação por partição, modelo persistido para
  inferência, referência congelada e testes em `tests/test_treinamento.py`.
- Contrato de qualidade de dados com Pandera (F1-03), aplicado às partições de
  treino, validação e teste a partir do manifesto da preparação, com relatório
  `quality.json`, bloqueio do conjunto inteiro em caso de violação e testes em
  `tests/test_validacao.py`. O manifesto registra também a semântica do alvo
  propagada desde a preparação e exigida pelo treinamento.
- Demonstração do bloqueio de ingestão de dados inválidos (F1-04), com geração
  de lote com anomalias intencionais, reprovação pelo contrato
  (`QUALIDADE_REPROVADA`) e testes em `tests/test_lote_invalido.py`.

- Simulação temporal de Data Drift (F2-01), com avanço do período e alteração
  intencional da distribuição de cinco features importantes, `features.csv`,
  rótulos em arquivo separado, manifesto com período e parâmetros, e testes em
  `tests/simulacao/data_drift`.

Pendentes:

- Confirmação da licença de uso do dataset.
- Fixação da versão remota do dataset e revisão das estatísticas anteriores
  para a amostra atual, conforme ressalvas em docs/DATASET.md.
- Definição dos demais cenários da produção simulada (controle e concept drift).
- Execução dos testes e Ruff no ambiente definido pelo `uv.lock` após estas
  correções; métricas e resultados históricos não foram reexecutados.
- Implementação das etapas restantes do pipeline (predição, detecção, dashboard).
