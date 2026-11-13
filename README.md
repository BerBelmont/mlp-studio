# MLP Studio

Trabalho acadêmico de Redes Neurais Artificiais: uma MLP com arquitetura
configurável, algoritmos matemáticos manuais e interface web em português.
Permite criar, preparar dados, treinar, avaliar, salvar, carregar e prever.
O mapa de conformidade dos requisitos 1.1–1.12 está em [REQUISITOS.md](REQUISITOS.md)
e as verificações de entrega estão em [AUDITORIA.md](AUDITORIA.md).

## Tecnologias e instalação

Python **3.12+**, NumPy, Pandas, Streamlit e Plotly. Pytest é a dependência de
desenvolvimento. JSON usa a biblioteca padrão. Não são utilizados TensorFlow,
Keras, PyTorch, Scikit-learn MLP ou diferenciação automática.

Execute na raiz do projeto, onde estão `app.py` e `pyproject.toml`:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run app.py
```

No Windows, use `py -3` em vez de `python3` e ative com
`.venv\Scripts\activate` no Prompt ou `.venv\Scripts\Activate.ps1` no PowerShell.
Abra o endereço exibido, normalmente `http://localhost:8501`. Para encerrar,
pressione Ctrl+C no terminal. Alternativa: `python -m streamlit run app.py`.
Sem os testes, a instalação pode usar `python -m pip install -e .`.
As dependências e a descoberta do pacote `src/mlp` estão em `pyproject.toml`.
Não é necessário editar caminhos nem configurar `PYTHONPATH`.

## Como usar a interface

1. **Dashboard:** oferece ações rápidas e apresenta a rede e os resultados
   existentes. Sem modelo ou histórico, mostra orientações em vez de métricas
   fictícias. As curvas e os números usam somente dados reais da sessão.
2. **Arquitetura Neural:** configure entradas, camadas ocultas (inclusive zero),
   neurônios, ativação de cada camada, saída e seed opcional. O editor ocupa
   35% e a prévia 65% da tela. A prévia muda imediatamente, antes de clicar em
   **Criar rede**, sem instanciar neurônios ou alterar o modelo ativo.
   Clique em **Criar rede** para aplicar. O desenho limita somente a exibição
   a 8 nós por camada e 10 camadas, sinalizando omissões. A rede real não tem
   essas limitações; a tabela contém todas as camadas. Contagens de neurônios
   excluem as entradas, e conexões correspondem aos pesos entre camadas.
3. **Dataset:** siga as três abas: **Importar CSV**, **Configurar dados** e
   **Conjuntos preparados**. Faça upload de um CSV numérico com cabeçalho e
   clique em **Carregar CSV**. Escolha colunas de entrada e saída na ordem
   desejada, percentuais, seed e normalização. **Validar e preparar conjuntos**
   verifica compatibilidade com a rede. Explore quantidades, proporções reais
   e as primeiras linhas de cada conjunto na terceira aba.
4. **Treinamento:** configure épocas, taxa de aprendizado e seed de
   embaralhamento na coluna esquerda. Clique em **Iniciar treinamento**.
   O painel à direita mostra status, curvas interativas de MSE e comparação
   inicial/final. Uma nova execução continua os pesos atuais e substitui o
   histórico exibido. Somente treino atualiza parâmetros. Validação avalia;
   teste não é passado à função de treinamento.
5. **Avaliação:** clique em **Avaliar conjunto de teste** após treinar. Veja o
   MSE final e a tabela esperado versus previsto. Não há atualização de pesos.
6. **Modelos e Previsões:** na aba **Modelos · JSON**, baixe a rede ou envie um
   JSON e clique em **Carregar rede JSON**. Na aba **Previsão manual**, informe
   valores originais na ordem de treinamento e clique em **Realizar previsão**.
   A rede aplica a normalização registrada exatamente uma vez. As saídas
   não são desnormalizadas, pois nunca foram normalizadas no treinamento.

Seeds podem ficar vazias para execução aleatória; preenchidas, devem ser
inteiros não negativos. A mesma arquitetura e as mesmas seeds de criação,
divisão e treinamento reproduzem os resultados no mesmo ambiente numérico.
Uma seed não garante convergência em qualquer dataset ou taxa de aprendizado.

## Dados e normalização

O CSV deve estar limpo, com atributos numéricos finitos. Não há imputação,
tratamento de categorias ou limpeza automática. A seleção rejeita colunas
inexistentes, repetidas, compartilhadas entre entrada/saída e dimensões
incompatíveis. Uma única saída continua sendo uma matriz com uma coluna.

A divisão embaralha **índices**, mantendo cada entrada associada à sua saída.
Percentuais positivos devem somar 100%, com tolerância absoluta de `1e-8`.
Treino e validação usam `floor(n * percentual / 100)`; o teste recebe o
restante. Assim, nenhum índice é perdido ou duplicado. Todos os conjuntos
precisam ser não vazios; datasets pequenos demais geram um erro claro.

A normalização é opcional: `none`, `minmax` ou `zscore`. Ocorre **depois da
divisão**, calculando parâmetros por coluna **exclusivamente nas entradas de
treino**. Validação, teste e novas previsões apenas reutilizam esses parâmetros.
Saídas esperadas permanecem intactas.

- Min-Max: `(x - minimo) / (maximo - minimo)`, sem limitar o resultado a [0, 1].
- Z-score: `(x - media) / desvio`, com desvio populacional (`ddof=0`).
- Atributo constante: divisor zero é substituído por 1 na transformação.
  O valor constante de treino torna-se zero; novos valores preservam sua
  diferença em relação à referência. Os parâmetros reais continuam armazenados.

## Exemplos para apresentar ao professor

Os dois CSVs estão em `data/`. Na interface, use **seed 42 na criação,
na divisão e no treinamento**, com divisão **70/15/15** em ambos os casos
(14 exemplos de treino, 3 de validação e 3 de teste).

| Configuração | Relação linear | XOR |
| --- | --- | --- |
| Arquivo | `data/linear.csv` | `data/xor.csv` |
| Entradas / saída | `x` / `y` | `x1`, `x2` / `y` |
| Arquitetura | 1 entrada → 1 saída Linear, sem ocultas | 2 entradas → 4 Tanh → 1 Sigmoid |
| Normalização | Min-Max | Nenhuma |
| Épocas | 200 | 1000 |
| Taxa de aprendizado | 0,05 | 0,3 |

A relação linear é `y = 2x + 1`, com 20 valores distintos de x entre -1 e 0,9.
O XOR contém os quatro padrões binários, repetidos cinco vezes para permitir
os três conjuntos não vazios. A divisão é por linha, portanto os mesmos
padrões podem aparecer nos três conjuntos: **isso demonstra aprendizado da
tabela XOR, não generalização para combinações inéditas**. Não há ruído nem
alteração dos valores 0 e 1. Para interpretar a saída Sigmoid como classe,
use o limiar 0,5 durante a apresentação; a aplicação retorna o valor contínuo.

Execute o ciclo completo sem interação, incluindo leitura CSV, treino,
avaliação final, JSON e comparação exata das previsões:

```bash
python examples/demonstracao.py
python examples/demonstracao.py --exemplo linear
python examples/demonstracao.py --exemplo xor
```

Resultados **obtidos na execução da auditoria** (Python 3.13.15):

| Exemplo | MSE treino inicial → final | MSE validação inicial → final | MSE teste final |
| --- | --- | --- | --- |
| Linear | 1,84747829 → 4,36295358e-17 | 1,66498297 → 7,79972252e-17 | 5,28295333e-17 |
| XOR | 0,266085698 → 0,000173855266 | 0,375092906 → 0,000216323341 | 0,000335654753 |

Previsões XOR observadas: `[0,0] → 0,009045`, `[0,1] → 0,978492`,
`[1,0] → 0,989774`, `[1,1] → 0,013000`. No linear, `[0,25] → 1,49999999813`.
As previsões foram exatamente iguais antes/depois de carregar o JSON, no
mesmo ambiente. Resultados podem variar entre versões e plataformas numéricas.

Roteiro de apresentação: execute o linear, mostre o gráfico e o teste,
preveja x=0,25 e baixe o JSON. Carregue-o em uma nova sessão e repita a
previsão. Depois crie uma nova rede para XOR, use as configurações da tabela
e mostre as quatro previsões. O dataset de teste não deve ser usado para
escolher hiperparâmetros; reserve-o para a avaliação final.

Os exemplos anteriores também foram preservados:

```bash
python examples/treinamento_simples.py
python examples/persistencia_simples.py
```

## Algoritmos implementados

`Neuron` armazena pesos, bias, z, saída e delta e calcula `z = x · w + bias`.
`Layer` reúne neurônios e `NeuralNetwork` encadeia camadas dinamicamente.
As ativações Linear, Sigmoid, Tanh e ReLU possuem derivadas em relação a z;
para ReLU em z=0, a derivada é zero. A Sigmoid evita overflow da exponencial.

Cada rede usa um gerador NumPy compartilhado durante a criação das camadas.
Xavier/Glorot normal usa desvio `sqrt(2/(fan_in + fan_out))` para Linear,
Sigmoid e Tanh; He normal usa `sqrt(2/fan_in)` para ReLU. Todos os biases
iniciam em zero e permanecem editáveis individualmente.

O forward passa as saídas de cada camada para a próxima e guarda z/saída.
O MSE é `soma((esperado - previsto)**2) / quantidade_total_de_valores`,
incluindo todas as saídas e exemplos. A avaliação não modifica parâmetros.

No backpropagation, para m saídas, `delta = (2/m) * (saida - esperado) * f'(z)`.
Nas ocultas, `delta = f'(z) * soma(peso_seguinte * delta_seguinte)`.
**Todos os deltas são calculados antes de qualquer atualização**. O gradiente
do peso é `delta * entrada`, e o do bias é `delta`. O SGD subtrai a taxa
multiplicada pelo gradiente após cada exemplo. A cada época, embaralha os
exemplos e calcula novamente o MSE completo de treino e validação somente
após todas as atualizações. Não há Adam, Early Stopping ou autodiferenciação.

APIs principais: `NeuralNetwork.add_layer`, `forward`, `train_step`,
`evaluate_mse`, `predict`; `treinamento.train`; `dados.load_csv`,
`select_columns`, `split_dataset`; `Normalizer.fit/transform`,
`normalize_splits`; `persistencia.save_network/load_network`.
`forward` recebe dados **já preparados**; `predict` recebe dados **originais**.
A API `train` altera a rede recebida; na interface, `train_session` treina em
uma cópia e confirma os parâmetros somente após sucesso. Falhas numéricas
preservam o modelo ativo e o histórico anterior, sem mudar as fórmulas.

## Formato JSON e previsão

Exemplo válido sem normalização:

```json
{
  "version": 1,
  "input_size": 1,
  "layer_count": 1,
  "layers": [{
    "neuron_count": 1,
    "activation": "linear",
    "weights": [[2.0]],
    "biases": [1.0]
  }],
  "normalization": null
}
```

`weights` tem uma linha por neurônio e uma coluna por entrada da camada;
`biases` segue a ordem dos neurônios. Ativações: `linear`, `sigmoid`, `tanh`,
`relu`. Arrays NumPy viram listas pelo módulo `json` nativo, sem pickle.

Com normalização, `normalization` é um objeto com `method`, `input_size`,
`minimum`, `maximum`, `mean`, `std`. Min-Max usa listas em minimum/maximum;
Z-score em mean/std. Os parâmetros não usados são `null`. O método `none`
ajustado pode ser salvo com os quatro parâmetros `null`.

O carregamento valida versão, contagens, ativações, formatos, números finitos
e normalizador; cria camadas diretamente com os parâmetros salvos, sem gerar
pesos aleatórios. O JSON não guarda histórico, z, saída, delta, nomes das
colunas ou estado dos geradores. Preserve a ordem e o significado dos atributos
originais; os rótulos após carregar serão Entrada 1, Entrada 2, etc.

## Estado, limites e organização

Rede ativa, rascunho, dataset, conjuntos, normalizador e resultados ficam na
sessão Streamlit. Navegar não apaga pesos. Criar/carregar uma rede ou substituir
o CSV invalida conjuntos e resultados antigos. Upload inválido mantém o estado
anterior. Configurações já confirmadas do dataset reaparecem ao retornar à página.
Uma rede treinada/carregada mantém seu normalizador ao preparar outro CSV;
para mudar a escala, crie uma nova rede. Refazer a divisão de dados de uma rede
já treinada não cria um teste independente do treinamento anterior.

A sessão não é armazenamento permanente: baixe o JSON antes de encerrar.
O treinamento é síncrono, com indicador de execução; redes grandes podem levar
mais tempo. Arquiteturas são configuráveis, sujeitas à memória do computador.
O desenho é simplificado para manter a legibilidade. Não há outro frontend.

```text
app.py                       entrada Streamlit, tema e navegação
src/mlp/neuronio.py           entidade Neuron
src/mlp/camada.py             entidade Layer e inicialização
src/mlp/rede.py               arquitetura, forward, SGD, avaliação e previsão
src/mlp/ativacoes.py          ativações e derivadas
src/mlp/metricas.py           MSE
src/mlp/treinamento.py        loop de épocas e histórico
src/mlp/dados.py              CSV, seleção e divisão
src/mlp/normalizacao.py       normalizador ajustado no treino
src/mlp/persistencia.py       JSON validado
src/mlp/interface.py          estado e adaptação das APIs à interface
src/mlp/ui/                   páginas e controles compartilhados
src/mlp/visual.py             cards, cabeçalhos e gráficos de resultados
src/mlp/architecture_view.py  desenho da rede e prévia do rascunho
src/mlp/ui_theme.py           tema dos gráficos
.streamlit/                  tema e CSS estático da aplicação
examples/                    demonstrações reproduzíveis
data/                       datasets de exemplo
models/                      local reservado a modelos salvos
tests/                      testes unitários, integrados e de interface
```

A CLI inicial foi mantida para compatibilidade (`python -m mlp`), com opções
informativas. A interface completa para o trabalho é o **Streamlit**.

## Testes e verificação

```bash
python -m pytest
python -m pip check
```

A suíte cobre fórmulas, gradient checking por diferenças finitas, sementes,
normalização sem vazamento, múltiplas arquiteturas/saídas, o fluxo completo de
CSV a JSON, exemplos linear/XOR e a interface via AppTest do Streamlit.
Também verifica preservação de parâmetros na avaliação/previsão, uploads,
download, navegação e atualização da prévia antes da criação da rede.
A interação automatizada com uploads exige uma versão de AppTest que a suporte;
as demais versões continuam cobertas pelos testes dos adaptadores de arquivo.

Para entrega, distribua a pasta do projeto com código, testes, documentação,
configuração, exemplos e `pyproject.toml`. Não inclua `.venv`, caches ou logs.
`.gitignore` já ignora esses arquivos; datasets e modelos não são ignorados.
