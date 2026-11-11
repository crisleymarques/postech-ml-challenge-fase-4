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

Essas verificações não validam o pipeline completo. As etapas de obtenção,
preparação, treinamento, validação, simulação (Data e Concept Drift), predição,
detecção e análise de drift, além dos logs, métricas e alertas e do dashboard,
possuem testes isolados descritos abaixo.

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

Os scripts de obtenção/amostragem do dataset e de preparação e separação das
partições estão implementados, assim como o contrato Pandera, o treinamento, a
simulação (Data e Concept Drift), a predição, a detecção e a análise estatística
do drift, a consolidação dos logs operacionais, as métricas de saúde, os alertas
e o dashboard Streamlit. A documentação de arquitetura
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

## Predição sobre os dados simulados de produção — F2-02

O componente [src/monitoramento/predicao/](src/monitoramento/predicao/)
carrega o pipeline do modelo baseline (F1-02) e o lote de features da simulação
(F2-01), aplica o pré-processamento já embutido no pipeline e gera as
predições para cada registro. A etapa recebe três manifestos de entrada
(treinamento, simulação e validação de produção), confere os hashes dos artefatos, valida que ambos
pertencem ao mesmo `dataset_id` e publica o resultado em
`data/analysis/<scenario_id>/<batch_id>/<model_id>/predictions.csv`, seguindo o
contrato da etapa de predição.

O pipeline inclui o pré-processamento ajustado no treino. O carregador também
mantém compatibilidade com artefatos antigos cujos conversores foram
serializados em `__main__`; novos modelos usam o módulo estável de
pré-processamento. Nenhum ajuste é realizado sobre o lote de produção.

```powershell
uv run python src/monitoramento/validacao.py ^
  --config configs/validacao.json --profile production ^
  --input-manifest reports/runs/<run_simulacao>/manifest.json ^
  --run-id run-f202-validacao

uv run gerar-predicoes ^
  --config configs/predicao.json ^
  --input-manifest reports/runs/<run_treinamento>/manifest.json ^
  --input-manifest reports/runs/<run_simulacao>/manifest.json ^
  --input-manifest reports/runs/run-f202-validacao/manifest.json ^
  --run-id run-f202-predicao
```

O arquivo de saída contém apenas `record_id`, `prediction` (classe prevista com
limiar 0,5) e `score` (probabilidade estimada da classe `Charged Off`), sem
revelar o rótulo. Na execução real sobre o cenário `data-drift` / lote
`batch-f201`, foram geradas 10.000 predições, com taxa de positivos de 57,32%
e score médio de 0,533.

```powershell
uv run pytest tests/predicao -q
```

Falhas da predição após o início publicam evento `failed` e manifesto terminal
com `status: failed`, sem saídas válidas, sempre que o diretório de execução
estiver gravável. O início é persistido antes da inferência; o manifesto é
publicado atomicamente por último. Configuração inválida e diretório de execução
já existente são recusados antes do início.

A predição exige SHA-256 hexadecimal de 64 caracteres para pipeline, metadados
e features. Ausência, formato inválido ou divergência bloqueiam o consumo.

A predição exige igualdade exata do `schema_version` do treinamento, dos
metadados e da simulação. Os metadados também devem corresponder ao
`dataset_id`, `model_id` e `reference_id` do manifesto do treinamento.

A aprovação de produção deve corresponder ao cenário, lote, dataset, schema,
produtor, caminho e hash exatos das features. O manifesto da predição registra
`quality_validation_run_id`. Chamadas antigas com apenas dois manifestos são
recusadas; execute a validação com `--profile production` e acrescente sua saída.

O diretório de predições de cada combinação cenário/lote/modelo é reservado
exclusivamente e nunca reutilizado, mesmo com outro `run_id`. O CSV é publicado
atomicamente. Uma colisão preserva resultados anteriores e gera execução falha.
Diretórios parciais de falhas ficam reservados para diagnóstico; uma nova
simulação deve usar novo `batch_id`, sem apagar resultados automaticamente.

## Detecção de Data Drift — F2-03

O componente [src/monitoramento/drift/deteccao/servico.py](src/monitoramento/drift/deteccao/servico.py)
compara a distribuição da referência congelada (features de treino) com a do
lote de produção simulado, usando o `DataDriftPreset` do Evidently descrito em
[configs/deteccao.json](configs/deteccao.json). A etapa recebe dois manifestos
de entrada (treinamento e simulação), confere os hashes dos artefatos e valida
que ambos pertencem ao mesmo `dataset_id`, reutilizando o contrato comum das
etapas anteriores. A referência é lida a partir dos metadados do modelo
(numéricas e categóricas) para aplicar os tipos corretos antes da comparação.

```powershell
uv run detectar-drift ^
  --config configs/deteccao.json ^
  --input-manifest reports/runs/<run_treinamento>/manifest.json ^
  --input-manifest reports/runs/<run_simulacao>/manifest.json ^
  --run-id run-f203-deteccao
```

| Artefato | Local |
|---|---|
| Resultado por variável | `reports/runs/<run_id>/drift.json` |
| Relatório HTML customizado com estatísticas do Evidently | `reports/runs/<run_id>/drift.html` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

Cada variável registra `feature`, `method`, `statistic`, `threshold` e
`drift_detected`. O Evidently escolhe o teste por tipo e tamanho da referência:
em referências maiores, usa distância de Wasserstein para numéricas e
Jensen-Shannon para categóricas; em amostras menores pode usar testes de
p-valor. A regra de decisão usa dois limiares declarados na configuração: `p_value_threshold`
para métodos de p-valor (drift quando a estatística fica abaixo) e
`distance_threshold` para métodos de distância (drift quando a estatística fica
acima), de modo que o limiar aplicado por variável fica explícito no resultado.
Variáveis não avaliáveis registram o motivo e `drift_detected` nulo, sem
converter ausência de evidência em ausência de drift. Um alerta de drift **não**
é erro: a execução termina em `success` e emite o evento `drift_alert` com nível
`WARNING`. A etapa aceita `--config`, `--input-manifest`, `--run-id` e
`--output-dir` e encerra com código `0` (sucesso) ou `1` (erro técnico ou
incompatibilidade de artefatos).

JSON, HTML customizado, métricas e alertas usam um único resultado canônico;
o HTML não reutiliza decisões nem limiares padrão do relatório nativo do
Evidently. Valores iguais ao limiar não indicam drift; o alerta de conjunto
exige proporção estritamente maior que 0,5, antes do arredondamento.

A detecção exige SHA-256 válido e correspondente para referência, lote e
metadados, e todas as features do modelo presentes nos dois CSVs. Colunas
inteiramente nulas são registradas como não avaliáveis e excluídas do
denominador; se nenhuma feature puder ser avaliada, a execução falha.
`changed_variables.not_detected` contém apenas variáveis avaliadas sem drift;
`not_evaluated` identifica as indisponíveis.

O serviço reserva o diretório exclusivamente, inclusive para chamadas Python
diretas e execuções concorrentes. O manifesto terminal é publicado por último,
por substituição atômica, após os relatórios e os eventos finais. Falhas retornam
1 e publicam diagnóstico `failed` quando a persistência está disponível; nunca
aprovam artefatos parciais. Uma publicação de sucesso precisa de manifesto e
eventos finais coerentes, incluindo `drift_alert` quando aplicável.

Os testes da detecção incluem regressões de cobertura, hashes obrigatórios,
concorrência, falhas de publicação e reprodução das decisões. O ambiente atual
emite avisos de depreciação de NumPy/Litestar nas dependências do Evidently;
são manutenção futura de compatibilidade, sem supressão global de avisos nem
atualização de versões sem falha demonstrada.

Na revalidação das correções, foi reutilizado o treinamento aprovado
`fase1-treinamento`, com referência congelada de **59.997 linhas**, da execução
sobre a base limpa. O baseline não foi retreinado nessa verificação. Um novo
lote de **10.000 linhas** passou por simulação, contrato de produção e predição.
Duas execuções independentes da detecção produziram os mesmos resultados:
**14/22 features com drift** (proporção **0,636364**), incluindo as cinco
alteradas intencionalmente, e evento `drift_alert`. Os hashes das partições
originais, da referência e dos artefatos do modelo permaneceram iguais.
Esses resultados usam a referência limpa e não substituem o registro histórico
abaixo, produzido com outra referência.

A suíte desta correção do F2-03 passou com **281 testes** (53 da detecção),
incluindo 10 verificações da consistência entre manifesto e eventos. O HTML
teve sua tabela e as decisões conferidas por parser nas regressões; a
inspeção visual no navegador não foi concluída porque o ambiente bloqueou
URLs de arquivos locais.

Na execução real sobre o cenário `data-drift` / lote `batch-f201`, 13 de 22
variáveis foram classificadas com drift (proporção de 0,5909, acima do critério
de conjunto de 0,5), incluindo as cinco alteradas pela simulação
(`annual_inc`, `dti`, `fico_range_low`, `fico_range_high` e `revol_util`), e o
evento `drift_alert` foi registrado.

```powershell
uv run pytest tests/drift/deteccao -q
```

## Análise estatística e interpretação do drift — F2-04

O componente [src/monitoramento/drift/analise/](src/monitoramento/drift/analise/)
complementa a detecção com testes estatísticos próprios e interpretação técnica,
descritos em [configs/analise_drift.json](configs/analise_drift.json). Para cada
variável o módulo calcula o **PSI** (Population Stability Index) e, no caso das
numéricas, o teste de **Kolmogorov-Smirnov** da SciPy. As amostras vêm dos mesmos
dois manifestos usados na detecção: a referência (features de treino) e o lote
simulado de produção, com verificação de hash e de `dataset_id`.

```powershell
uv run analisar-drift ^
  --config configs/analise_drift.json ^
  --input-manifest reports/runs/<run_treinamento>/manifest.json ^
  --input-manifest reports/runs/<run_simulacao>/manifest.json ^
  --run-id run-f204-analise
```

| Artefato | Local |
|---|---|
| Resultado por variável e interpretação | `reports/runs/<run_id>/analise.json` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

A execução reserva um diretório novo e publica o manifesto atomicamente por
último, após os resultados e eventos finais. Falhas de entrada, cálculo ou IO
retornam código 1 e, quando a persistência está disponível, publicam configuração,
eventos, métricas vazias e manifesto `failed` com `outputs: []`. Artefatos parciais
não são consumíveis. Configuração inválida e destino existente são recusados
antes do início; reexecuções não sobrescrevem publicações anteriores.

Todas as features dos metadados devem existir na referência e no lote.
Uma coluna ausente bloqueia a análise, identifica a entrada incompatível e
não produz um relatório parcial nem uma confirmação estatística fictícia.

PSI e K-S são avaliados independentemente. Quando os quantis da referência
não permitem calcular PSI, o K-S continua válido para a decisão. Apenas a
indisponibilidade de todos os testes deixa `drift_detected` nulo; nesse caso,
a variável aparece como não avaliável, sem entrar no denominador do resumo.

Cada variável registra `psi`, `psi_band` (insignificante, moderada ou
significativa), `ks_statistic`, `ks_p_value`, `drift_detected` e `reasons`. A
regra de decisão combina os dois testes: há mudança quando o PSI atinge
`psi_threshold` **ou** quando o p-valor do K-S fica abaixo de
`ks_p_value_threshold` e a estatística atinge `ks_statistic_threshold`. O
documento cruza as variáveis alteradas intencionalmente na simulação
(`changed_variables.configured`) com as confirmadas pelos testes
(`confirmed`/`not_confirmed`) e registra conclusões técnicas sem inferir
causalidade. Como na detecção, um alerta **não** é erro: a execução termina em
`success` e emite `drift_alert` com nível `WARNING`. A etapa aceita `--config`,
`--input-manifest`, `--run-id` e `--output-dir` e encerra com código `0`
(sucesso) ou `1` (erro técnico ou incompatibilidade de artefatos).

Na execução real sobre o cenário `data-drift` / lote `batch-f201`, 8 de 22
variáveis foram classificadas com mudança (proporção de 0,3636, PSI máximo de
0,7195), e as cinco variáveis alteradas pela simulação (`annual_inc`, `dti`,
`fico_range_low`, `fico_range_high` e `revol_util`) foram confirmadas. O
documento também registra a ressalva de que referência e lote provêm de períodos
distintos, de modo que parte das diferenças pode refletir a separação temporal e
não apenas a alteração injetada.

Na revalidação das correções do PR #37, usando a referência aprovada de
59.997 linhas e o lote `batch-pr36` de 10.000 linhas já usados na validação
do #36, duas análises independentes produziram os mesmos resultados:
8/22 features com mudança (0,363636), PSI máximo 0,749836 e as cinco features
alteradas confirmadas. Ambas emitiram `drift_alert` e terminaram em `success`.
Os hashes dos dados e dos artefatos do modelo permaneceram inalterados, sem
retreinar o baseline. Esses números correspondem a outra referência/lote
em relação à execução histórica acima.

```powershell
uv run pytest tests/drift/analise -q
```

## Simulação e análise de Concept Drift — F2-05

O Concept Drift é a mudança da relação entre as entradas e o alvo (P(y|X)) — e
não da distribuição das entradas (P(X)), que é o Data Drift. A F2-05 tem dois
componentes: um gera o cenário a partir da base reservada de produção e o outro
compara o desempenho do modelo entre o conceito original e o cenário.

O componente
[src/monitoramento/simulacao/concept_drift/](src/monitoramento/simulacao/concept_drift/)
mantém as features inalteradas e altera apenas o desfecho de um segmento. No
cenário de recessão, tomadores `prime` (`fico_range_low >= 700`), historicamente
considerados seguros, passam a inadimplir mais do que o histórico projetava.
Somente rótulos negativos podem virar positivos, com probabilidade crescente com
o FICO e limitada por `max_probability`, escolhidos com semente fixa para
reprodutibilidade, conforme [configs/simulacao_concept.json](configs/simulacao_concept.json).

| Artefato | Local |
|---|---|
| Features inalteradas | `data/analysis/<scenario_id>/<batch_id>/features.csv` |
| Rótulos do novo conceito | `data/analysis/<scenario_id>/<batch_id>/labels.csv` |
| Rótulos originais (contraste) | `data/analysis/<scenario_id>/<batch_id>/labels_reference.csv` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

```powershell
uv run simular-concept-drift ^
  --config configs/simulacao_concept.json ^
  --batch-id batch-f205 ^
  --run-id run-f205-concept
```

Antes da avaliação, aprove as features do cenário pelo contrato de produção:

```powershell
uv run python src/monitoramento/validacao.py ^
  --config configs/validacao.json ^
  --profile production ^
  --input-manifest reports/runs/<run_concept_drift>/manifest.json ^
  --run-id <run_validacao_concept>
```

O componente
[src/monitoramento/avaliacao/concept_drift/](src/monitoramento/avaliacao/concept_drift/)
aplica o pipeline do treinamento sobre as features (inalteradas) e calcula as
métricas contra os dois conjuntos de rótulos — referência e novo conceito —,
seguindo [configs/avaliacao_concept.json](configs/avaliacao_concept.json). Como
os registros e as features são os mesmos, apenas a relação perfil–desfecho muda,
o que isola a mudança de conceito da mudança de distribuição.

```powershell
uv run avaliar-concept-drift ^
  --config configs/avaliacao_concept.json ^
  --input-manifest reports/runs/<run_treinamento>/manifest.json ^
  --input-manifest reports/runs/<run_concept_drift>/manifest.json ^
  --input-manifest reports/runs/<run_validacao_concept>/manifest.json ^
  --run-id run-f205-avaliacao
```

| Artefato | Local |
|---|---|
| Métricas por conceito, comparação e interpretação | `reports/runs/<run_id>/performance.json` |
| Manifesto, eventos, métricas e config | `reports/runs/<run_id>/` |

A degradação é declarada na configuração (`roc_auc`, queda mínima `min_delta`).
Quando a variação atinge o critério, a execução registra `degradation_detected` e
emite `drift_alert` com nível `WARNING`, terminando ainda em `success` — um
alerta não é erro. O documento `performance.json` deixa explícita a diferença
entre Data Drift, Concept Drift e queda de desempenho: mudar a relação P(y|X) não
implica, por si só, queda — a degradação é uma consequência possível, não a
definição. A etapa usa `stage: avaliacao` e aceita `--config`,
`--input-manifest` (repetível), `--run-id` e `--output-dir`.

A avaliação exige aprovação de qualidade vinculada ao mesmo dataset, schema,
cenário, lote, produtor e hash das features. Modelo e metadados também têm
identidade, schema e integridade conferidos antes da inferência. Simulação e
avaliação rejeitam rótulos fora de 0/1, frações, ausências e valores não finitos,
sem truncamento; os IDs dos rótulos devem corresponder exatamente ao lote.

Diretórios de execução e lotes são reservados exclusivamente, inclusive em
chamadas Python e processos concorrentes. Reexecuções exigem novos IDs e
preservam os resultados anteriores. CSVs e JSONs são publicados atomicamente;
eventos finais precedem o manifesto terminal. Falhas após o início geram
`failed`, métricas vazias e manifesto sem saídas aprovadas quando a persistência
estiver disponível. Configuração inválida não inicia uma execução.

Quando a métrica de comparação não puder ser calculada (por exemplo, ROC AUC
com uma única classe), `metric_delta` e `degradation_detected` são nulos e
`availability` é `not_available`. A execução termina controladamente, sem
alerta de degradação e sem interpretar falta de evidência como ausência de queda.

Na revalidação das correções, o fluxo simulação → validação de produção →
avaliação foi executado duas vezes sobre 10.000 registros e o baseline já
persistido. Houve 2.032 rótulos alterados; ROC AUC de 0,671789 para 0,449332
(variação −0,222457), com `drift_alert`. As duas execuções produziram os mesmos
hashes de features/rótulos e métricas. Os hashes das partições originais e do
modelo permaneceram inalterados. Colisão de lote e ausência de aprovação foram
bloqueadas. Esses números usam outra amostra/referência e não substituem a
execução histórica descrita abaixo.

Na execução real sobre o cenário `concept-drift` / lote `batch-f205`, 4.890
registros caíram no segmento e 1.916 rótulos foram alterados (taxa de positivos
de 0,195 → 0,3866). O `roc_auc` caiu de 0,6825 (conceito original) para 0,4612
(variação −0,2213), evidenciando degradação associada à mudança de conceito, com
evento `drift_alert` registrado.

```powershell
uv run pytest tests/simulacao/concept_drift tests/avaliacao/concept_drift -q
```

## Logs operacionais centralizados — F3-01

O componente [src/monitoramento/logs.py](src/monitoramento/logs.py) consolida os
`events.jsonl` que cada etapa publica em `reports/runs/<run_id>/`, seguindo o
contrato comum de evento. Cada evento é classificado por categoria operacional
(ingestão, transformação, validação, treinamento, simulação, predição, drift e
avaliação) e por resultado (início, sucesso, rejeição, falha, alerta e
ignorado), sem alterar as etapas existentes — apenas agrega o que já é emitido.

```powershell
uv run python src/monitoramento/logs.py --runs-dir reports/runs --output-dir reports/logs
```

| Artefato | Local |
|---|---|
| Log operacional consolidado (todos os eventos) | `reports/logs/operacional.jsonl` |
| Resumo com contagens, falhas e alertas | `reports/logs/resumo.json` |

O resumo reúne o total de execuções e a contagem por status
(`success`/`rejected`/`failed`/`incomplete`), a contagem por categoria e
resultado, a lista de **falhas** com `run_id`, `stage` e `error_code` — o que
torna as violações de contrato identificáveis — e a lista de **alertas** de
drift com `scenario_id` e `batch_id`. A etapa `preparacao` realiza ingestão e
transformação de forma acoplada; por isso seu desfecho alimenta as duas
categorias e as transformações aplicadas são detalhadas a partir do manifesto.
Manifestos com status `failed` ou `rejected` também alimentam a lista de falhas
quando não há evento terminal equivalente, preservando o código e o horário do
erro sem duplicar registros ou criar eventos no log original. As demais execuções
sem publicação terminal confirmada aparecem como `incomplete`, sem inferir sucesso.
Os artefatos ficam em `reports/logs/`, ignorado pelo Git, e a etapa encerra com
código `0` (sucesso) ou `1` (erro técnico).

Na execução real sobre as 11 execuções registradas, o log consolidou 25 eventos
(10 sucessos e 1 rejeição), 1 falha de contrato (`QUALIDADE_REPROVADA`) e 3
alertas de drift.

```powershell
uv run pytest tests/test_logs.py -q
```

## Métricas de saúde operacional — F3-02

O catálogo de métricas está definido em
[configs/metricas_saude.json](configs/metricas_saude.json) e rege a coleta. Cada
métrica declara `id`, `categoria`, `nome`, `semantica`, `unidade`, `escopo`
(`execucao` ou `agregado`), `etapas` (nulo quando aplicável a todas), `coleta`,
`criterio`, `direcao` (`maior_pior`/`menor_pior`) e os limiares
`limiar_degracao` e `limiar_falha`. São 15 métricas em 5 categorias: pipeline,
qualidade, drift, modelo e infraestrutura local.

O componente [src/monitoramento/metricas_saude.py](src/monitoramento/metricas_saude.py)
lê os artefatos persistidos de cada execução — manifesto, `metrics.json`,
`quality.json`, `drift.json`, `analise.json` e `performance.json` —, coleta os
valores, marca a disponibilidade e classifica cada medida em `ok`, `atencao`,
`critico`, `indisponivel` ou `nao_aplicavel`. A sinalização e o alerta efetivos
ficam na integração do F3-03; esta etapa apenas coleta e classifica.

```powershell
uv run python src/monitoramento/metricas_saude.py --runs-dir reports/runs --output-dir reports/saude
```

| Artefato | Local |
|---|---|
| Medidas por execução, agregados e resumo | `reports/saude/saude.json` |

Os limiares operacionais definidos são: duração 60 s (atenção) e 300 s
(crítico); taxa de sucesso 0,9 (atenção) e 0,5 (crítico); proporção de variáveis
com drift 0,1 (atenção) e 0,5 (crítico); `roc_auc` 0,6 (atenção) e 0,5
(crítico); espaço livre 1 GiB (atenção) e 256 MiB (crítico). Para métricas
`menor_pior`, os limites são exclusivos: taxa de sucesso 0,9 e ROC AUC 0,6
ainda são `ok`; ROC AUC 0,5 é `atencao`. Volume de ao menos 1 linha é `ok`.
Para `maior_pior`, o próprio limiar já aciona a situação correspondente.
O relatório é publicado atomicamente: leitores recebem um JSON completo e
uma falha na atualização preserva a última versão publicada. Alerta de drift e
mudança de distribuição não são erro de execução. Os artefatos ficam em
`reports/saude/`, ignorado pelo Git, e a etapa encerra com código `0` (sucesso)
ou `1` (erro técnico).

Na execução real sobre as 11 execuções registradas, foram coletadas 48 medidas
(situações `ok=38`, `atencao=8`, `critico=2`), com destaque para
`drift.proporcao_drift = 0,590909` (`run-f203-deteccao`) e
`modelo.roc_auc = 0,461174` (`run-f205-avaliacao`) em estado crítico.

```powershell
uv run pytest tests/test_metricas_saude.py -q
```

## Integração de drift, métricas e alertas — F3-03

O componente [src/monitoramento/alertas.py](src/monitoramento/alertas.py) integra
os artefatos já consolidados — as métricas de saúde (`reports/saude/saude.json`,
F3-02) e os logs operacionais (`reports/logs/resumo.json`, F3-01) — e aplica os
critérios declarados em [configs/alertas.json](configs/alertas.json):

- medida de saúde com situação `atencao` gera alerta `warning`;
- medida de saúde com situação `critico` gera alerta `critical`;
- falha de execução ou de contrato registrada nos logs gera alerta `critical`;
- evento `drift_alert` registrado nos logs gera alerta `warning`.

```powershell
uv run python src/monitoramento/alertas.py --saude reports/saude/saude.json --logs-resumo reports/logs/resumo.json --config configs/alertas.json --output-dir reports/alertas
```

| Artefato | Local |
|---|---|
| Alertas, bloco de drift e resumo | `reports/alertas/alertas.json` |

Cada alerta é identificável (`id`, `severidade`, `categoria`, `origem`, `fonte`,
`valor`, `situacao`, `error_code` e `evidencia`) e contabilizável. O artefato
reúne ainda um bloco consolidado de drift (presença, variáveis com drift,
proporção e total de alertas) e um resumo por severidade, categoria e origem.
O bloco `drift.por_execucao` preserva os resultados de cada execução. Os campos
de destaque (`dataset_drift`, `variaveis_com_drift` e `proporcao_drift`) pertencem
à mesma `execucao_resumo`: a de maior proporção, com desempate por quantidade
e `run_id`. Esse destaque é histórico, não o resultado mais recente; a análise
estatística não fornece `dataset_drift`, que permanece nulo nesse caso. O total
de alertas continua agregado sobre todas as execuções.

A configuração exige as situações `atencao` e `critico`, mapeadas para as
severidades suportadas (`warning` e `critical`). Antes da publicação, o componente
valida a versão principal e a estrutura dos contratos de saúde e logs, incluindo
as medidas e os eventos consumidos. Arquivos vazios (`{}`), coleções malformadas
e versões incompatíveis encerram com código `1`, sem criar ou sobrescrever
`alertas.json`. Coleções legitimamente vazias em contratos válidos são aceitas.

O componente apenas integra e sinaliza: não relê os artefatos brutos, não altera
as etapas anteriores e não converte alerta em erro — encerra com código `0`
(sucesso) ou `1` (erro técnico). Os artefatos ficam em `reports/alertas/`,
ignorado pelo Git.

Na execução real sobre os relatórios consolidados, foram integrados 14 alertas
(`warning=11`, `critical=3`), com o bloco de drift indicando `dataset_drift=True`,
13 variáveis com drift, proporção 0,590909 e 3 alertas de drift.

```powershell
uv run pytest tests/test_alertas.py tests/test_alertas_regressoes.py -q
```

## Dashboard de monitoramento — F3-04

O componente [src/monitoramento/dashboard.py](src/monitoramento/dashboard.py) é o
painel Streamlit que reúne os resultados persistidos pelas etapas anteriores. Ele
lê os eventos e manifestos de cada execução em `reports/runs/<run_id>/` e os
artefatos consolidados — logs operacionais (F3-01), métricas de saúde (F3-02) e
alertas integrados (F3-03) — sem executar treino nem alterar dados.

```powershell
uv run streamlit run src/monitoramento/dashboard.py
```

Os caminhos das fontes são configuráveis na barra lateral; por padrão:
`reports/runs`, `reports/logs/resumo.json`, `reports/saude/saude.json` e
`reports/alertas/alertas.json`.

| Aba | Conteúdo |
|---|---|
| Visão geral | Execuções, taxa de sucesso, falhas, rejeições, alertas, saúde dos artefatos e histórico por etapa |
| Saúde operacional | Medidas de saúde por execução e agregados, filtráveis por categoria (F3-02) |
| Drift | Destaque histórico identificado por execução, resultados individuais e total de alertas (F3-03) |
| Alertas | Alertas por severidade, categoria e origem, com filtros (F3-03) |
| Logs e falhas | Contagens de execuções e eventos por categoria, além das falhas registradas (F3-01) |

O painel é tolerante a ausências: artefato inexistente aparece como
`indisponivel` sem impedir as demais abas, e execução sem evento terminal é
apresentada como incompleta, sem inferir sucesso.

Eventos com JSON inválido ou valores que não sejam objetos são ignorados.
Falhas de acesso ou de codificação de um arquivo de eventos ficam isoladas à
execução, sem impedir as demais abas nem confirmar sucesso por leitura parcial.
Relatórios de alertas ausentes, ilegíveis ou sem resumo e coleção utilizáveis
aparecem como indisponíveis; seus indicadores exibem `—`. Zero é exibido apenas
quando confirmado por um relatório válido.

A [revalidação dos critérios da F3-04](docs/VALIDACAO_F3_04.md) registra as
evidências, a captura já publicada no PR e o acesso local ao painel.

```powershell
uv run pytest tests/test_dashboard.py tests/test_dashboard_resiliencia.py tests/test_alertas_dashboard_regressoes.py -q
```

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
- Predição sobre os dados simulados de produção (F2-02), reutilizando o modelo
  baseline e o lote da simulação, com pré-processamento consistente com o
  treinamento, `predictions.csv` em `data/analysis/` e testes em
  `tests/predicao`.
- Detecção de Data Drift com Evidently (F2-03), comparando a referência
  congelada com o lote de produção simulado, com regra de decisão por dois
  limiares, resultado por variável, relatório `drift.json`/`drift.html`, evento
  `drift_alert` sem converter alerta em erro e testes em
  `tests/drift/deteccao`.
- Análise estatística e interpretação do drift (F2-04), com PSI e teste de
  Kolmogorov-Smirnov por variável, cruzamento com as variáveis alteradas pela
  simulação, conclusões técnicas em `analise.json` e testes em
  `tests/drift/analise`.
- Simulação e análise de Concept Drift (F2-05), com cenário de mudança de
  relação entre perfil e desfecho (P(y|X)) mantendo as features inalteradas,
  comparação de desempenho entre conceito original e cenário em
  `performance.json`, evento `drift_alert` sem converter alerta em erro e testes
  em `tests/simulacao/concept_drift` e `tests/avaliacao/concept_drift`.

- Consolidação dos logs operacionais (F3-01), com classificação dos eventos por
  categoria e resultado, resumo de sucessos, falhas de contrato e alertas em
  `reports/logs/` e testes em `tests/test_logs.py`.
- Definição e coleta das métricas de saúde operacional (F3-02), com catálogo de
  15 métricas em 5 categorias, coleta a partir dos artefatos persistidos,
  classificação por situação com dois níveis de limiar (`atencao`/`critico`),
  medidas em `reports/saude/saude.json` e testes em
  `tests/test_metricas_saude.py`.
- Integração de drift, métricas e alertas (F3-03), com critérios de alerta em
  `configs/alertas.json`, sinalização de falhas e degradação com severidade,
  bloco consolidado de drift, alertas identificáveis e contabilizáveis em
  `reports/alertas/alertas.json` e testes em `tests/test_alertas.py`.
- Dashboard de monitoramento (F3-04), com painel Streamlit em abas por tema
  (visão geral, saúde operacional, drift, alertas e logs), fontes configuráveis
  na barra lateral, tratamento de artefatos ausentes como `indisponivel` e de
  execuções sem evento terminal como incompletas, e testes das funções de carga
  e agregação em `tests/test_dashboard.py`.

Pendentes:

- Confirmação da licença de uso do dataset.
- Fixação da versão remota do dataset e revisão das estatísticas anteriores
  para a amostra atual, conforme ressalvas em docs/DATASET.md.
- Definição do cenário de controle da produção simulada.
- Execução dos testes e Ruff no ambiente definido pelo `uv.lock` após estas
  correções; métricas e resultados históricos não foram reexecutados.
- Validação funcional integrada do pipeline com dados sintéticos, além dos
  testes isolados de obtenção, de preparação do dataset, de treinamento e de
  validação.
