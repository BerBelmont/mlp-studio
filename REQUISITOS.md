# Auditoria dos requisitos acadêmicos

Classificação final após revisão do código, correções e execução dos testes:
**12 ATENDIDOS, 0 PARCIALMENTE ATENDIDOS, 0 NÃO ATENDIDOS**.

| Requisito | Classificação | Arquivos e funções | Evidência e testes |
| --- | --- | --- | --- |
| 1.1 — Neurônio | ATENDIDO | `src/mlp/neuronio.py`: `Neuron.__init__`, `forward` | Pesos, bias, z, output e delta por neurônio; produto NumPy e ativação. `tests/test_neuronio.py`. |
| 1.2 — Camada | ATENDIDO | `src/mlp/camada.py`: `Layer.__init__`, `forward` | Quantidade configurável, ativação compartilhada e saída vetorial; dimensões verificadas. `tests/test_camada.py`. |
| 1.3 — Rede dinâmica | ATENDIDO | `src/mlp/rede.py`: `NeuralNetwork.__init__`, `add_layer` | Entrada configurável e camadas adicionadas dinamicamente; aceita só saída ou múltiplas ocultas e saídas. `tests/test_rede.py`, `test_integracao.py`. |
| 1.4 — Ativações | ATENDIDO | `src/mlp/ativacoes.py`: `linear`, `sigmoid`, `tanh`, `relu`, respectivas `_derivative`, `DERIVATIVES` | Quatro funções e derivadas em z; ReLU'(0)=0 e Sigmoid estável. Valores conhecidos e diferenças finitas em `tests/test_ativacoes.py`. |
| 1.5 — Inicialização | ATENDIDO | `Layer.__init__`, `NeuralNetwork.__init__`, `add_layer` | Xavier normal `sqrt(2/(fan_in+fan_out))`, He normal `sqrt(2/fan_in)` para ReLU; biases zero; único Generator por rede. `tests/test_inicializacao.py`. |
| 1.6 — Forward | ATENDIDO | `Neuron.forward`, `Layer.forward`, `NeuralNetwork.forward` | Validação do vetor, encadeamento dinâmico, ativação por camada e estado intermediário armazenado. Sem alterar parâmetros. `tests/test_forward.py`. |
| 1.7 — MSE | ATENDIDO | `src/mlp/metricas.py`: `mse`; `NeuralNetwork.evaluate_mse` | Soma manual dos quadrados dividida pelo total de valores; vários exemplos e saídas; rejeita vazios e formatos incompatíveis. `tests/test_metricas.py`, `test_avaliacao.py`. |
| 1.8 — Backpropagation/SGD | ATENDIDO | `NeuralNetwork.train_step` | Regra da cadeia com fator 2/m na saída, deltas ocultos em ordem reversa; todos os deltas antes das atualizações. Gradientes delta*entrada e delta para bias; atualização por exemplo. `tests/test_backpropagation.py` inclui gradient checking de todos os pesos/biases em rede com duas ocultas. |
| 1.9 — Dados e normalização | ATENDIDO | `src/mlp/dados.py`: `load_csv`, `select_columns`, `split_dataset`; `src/mlp/normalizacao.py`: `Normalizer.fit`, `transform`, `normalize_splits` | Pandas/CSV, colunas disjuntas e compatíveis, divisão por índices com percentuais e seed; nenhuma perda/duplicação de índices. Divisão antes da escala; parâmetros somente no treino; saídas intactas; none/Min-Max/Z-score. `tests/test_dados.py`, `test_normalizacao.py`, `test_integracao.py`. |
| 1.10 — Treinamento | ATENDIDO | `src/mlp/treinamento.py`: `train`; `src/mlp/interface.py`: `train_session` | SGD para cada índice embaralhado por época, depois MSE completo de treino/validação; histórico por época. Teste não é argumento de train. Interface confirma os parâmetros apenas após sucesso. `tests/test_treinamento.py`, `test_integracao.py`, `test_auditoria.py`. |
| 1.11 — JSON | ATENDIDO | `src/mlp/persistencia.py`: `save_network`, `load_network`, `_restore`; `Layer.__init__` com pesos/biases fornecidos | Arquitetura, contagens, nomes das ativações, parâmetros e normalizador serializados; validação e restauração sem gerar novos pesos. Previsões exatamente iguais antes/depois. `tests/test_persistencia.py`, `test_integracao.py`. |
| 1.12 — Previsão | ATENDIDO | `NeuralNetwork.predict`; `src/mlp/ui/models.py`: `prediction_page` | Vetor original validado; normalizador registrado aplicado uma vez; somente forward, sem desnormalizar saída nem atualizar pesos/biases. `tests/test_persistencia.py`, `test_integracao.py`, `test_app.py`. |

## Restrições verificadas

- Nenhuma importação de biblioteca pronta de redes neurais ou autodiferenciação.
- Forward, MSE, backpropagation e SGD escritos com operações NumPy explícitas.
- Quantidades e ativações são parâmetros: os números nos exemplos e defaults
  da interface não fixam a arquitetura do pacote.
- Todas as operações do enunciado estão acessíveis pelo Streamlit, sem editar
  o código: `src/mlp/ui/architecture.py`, `dataset.py`, `training.py`, `models.py`.
- Separação entre entidades/rede, métricas, treinamento, dados, normalização,
  persistência e interface. A CLI inicial é informativa e não substitui o Streamlit.
- Normalização nunca é ajustada dentro de `train` ou `predict`.
- Validação e teste usam avaliação, sem cálculo de gradientes.
- Não há Adam, RMSProp, Early Stopping, pickle ou frontend separado.

## Evidência integrada

`tests/test_integracao.py` executa 12 combinações do ciclo completo, cruzando
none/Min-Max/Z-score, só camada de saída/duas ocultas e uma/duas saídas.
Verifica CSV, seleção, divisão, escala, treino, MSE, teste, JSON e previsões,
incluindo a imutabilidade dos parâmetros durante avaliação e previsão.
Também cobre dados incompatíveis, JSON inválido e os exemplos entregues,
executados a partir de outro diretório.

`tests/test_app.py` cobre navegação, prévia reativa, preservação do modelo
treinado, preparação, treinamento, avaliação, uploads CSV/JSON e download.
`tests/test_auditoria.py` cobre rejeição de valores não finitos e preservação
do modelo/histórico em falha numérica na interface.

## Limites explícitos, fora dos requisitos

Não há limpeza de dados nem codificação de categorias. O JSON não persiste
histórico, estado do gerador ou nomes de atributos; a ordem original deve ser
preservada. XOR repete os quatro padrões para permitir a divisão em três
conjuntos: os resultados demonstram aprendizado, não generalização inédita.
