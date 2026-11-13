# Relatório de auditoria e validação para entrega

Auditoria executada em 13/11/2026. Escopo: requisitos 1.1–1.12, restrições
acadêmicas, fluxo integrado, interface, exemplos, documentação e instalação.
O projeto e os algoritmos existentes foram preservados.

## Resultado

- **324 testes Pytest aprovados**, nenhum teste ignorado nesta execução.
- **12 requisitos ATENDIDOS, 0 PARCIALMENTE ATENDIDOS e 0 NÃO ATENDIDOS**.
- Evidências individuais: [REQUISITOS.md](REQUISITOS.md).
- Nenhuma biblioteca pronta de redes neurais, autodiferenciação ou pickle
  encontrada nas importações do código-fonte.

## Problemas encontrados e correções

| Problema | Correção | Verificação |
| --- | --- | --- |
| A API direta podia aceitar NaN/Inf em pesos, entradas, saídas ou MSE, apesar de a leitura CSV já rejeitá-los. | Validação explícita de valores finitos em Neuron, train_step, mse e split_dataset. Não mudou nenhuma fórmula. | `tests/test_auditoria.py` e testes anteriores. |
| Taxa complexa ou percentual não numérico podia produzir erro pouco claro. | Validação de taxa real finita positiva e mensagem em português para percentuais inválidos. | `tests/test_auditoria.py`. |
| A interface treinava o modelo ativo diretamente e descartava o histórico antes de saber se a execução seria válida. Uma divergência podia deixar parâmetros inutilizáveis. | `interface.train_session` treina em cópia, detecta erros numéricos e confirma parâmetros/estado somente no sucesso, preservando a identidade da rede ativa. Usa integralmente `treinamento.train`. | Teste de divergência com taxa extrema; pesos, bias, estado, histórico e conjuntos anteriores preservados. |
| Ao navegar, o formulário do dataset podia mostrar seleção/percentuais padrão diferentes dos conjuntos já preparados. | Estado `preparation_config` e restauração de colunas, percentuais, normalização e seed confirmados. Substituição de CSV/rede invalida essas configurações derivadas. | Fluxo AppTest com upload CSV, navegação, treino, download e upload JSON. |
| Documentação extensa de etapas anteriores e falta de CSVs para apresentação. | README consolidado, mapa de requisitos, este relatório, datasets linear/XOR e demonstração executável. | Exemplos realmente executados na instalação original e em ambiente isolado. |

As alterações são de validação, integração e documentação. Forward,
backpropagation, SGD, MSE, inicialização e normalização mantêm suas fórmulas.
O formato JSON versão 1 foi preservado. Não foi realizado outro redesign.

## Testes e verificações executados

| Verificação | Resultado |
| --- | --- |
| `python -m pytest -q` | 324 aprovados, incluindo gradient checking e 12 combinações do fluxo completo. |
| `python -m pip check` | Nenhuma dependência incompatível, tanto no ambiente de desenvolvimento quanto no ambiente isolado. |
| Importação de módulos com `pkgutil`/`importlib` | Todos os 22 módulos do pacote importados. |
| AppTest do Streamlit | Navegação, editor reativo, treinamento, avaliação, previsão, uploads CSV/JSON e download executados sem exceções. |
| Inicialização real do servidor Streamlit | Servidor iniciado em loopback; `/_stcore/health` retornou HTTP 200 e `ok`. |
| `examples/treinamento_simples.py` | Executado; reduziu MSE de treino/validação. |
| `examples/persistencia_simples.py` | Executado; previsão `1.499999998822` exatamente igual antes/depois de JSON. |
| `examples/demonstracao.py` | Linear e XOR executados, com teste apenas ao final e previsões idênticas após JSON. |
| Construção com `pip wheel --no-deps` | Wheel construído com sucesso a partir de `pyproject.toml`. |
| Instalação limpa do wheel | Instalado com suas dependências em um novo venv temporário, sem reutilizar a `.venv` do projeto. |
| Cópia portátil sem `src/` | app/config/datasets/exemplos copiados para outro diretório; 22 módulos importados do wheel instalado. Interface inicializada e rede criada via AppTest; ambas as demonstrações reproduzidas. |
| Caminhos e bibliotecas | Nenhum caminho absoluto fixo do desenvolvedor e nenhuma importação de biblioteca pronta de redes neurais/autodiferenciação em código ou exemplos. |
| `.gitignore` | `.venv`, `__pycache__`, bytecode, caches Pytest, metadados de instalação, build/dist, logs e temporários ignorados. CSVs, testes e modelos preservados. |

O sandbox bloqueou inicialmente a obtenção das dependências de build pelo
PyPI. A mesma operação foi repetida com a permissão de rede necessária e
concluída. Isso foi uma limitação do ambiente, não uma falha do projeto.

## Ambiente usado

Python 3.13.15; NumPy 2.5.4; Pandas 3.1.1; Streamlit 1.67.0; Plotly 6.9.0;
Pytest 9.2.0. O projeto declara Python 3.12+; Python 3.12 e Windows não foram
executados nesta máquina. A verificação da interface foi feita por AppTest
e inicialização/saúde HTTP do servidor, sem inspeção manual em navegador de
todos os tamanhos de tela. Não se declara essa inspeção como realizada.

## Resultados de demonstração observados

Seeds 42 para inicialização, divisão e treinamento; divisão 70/15/15.
Valores arredondados apenas para apresentação:

| Exemplo | MSE treino inicial | MSE treino final | MSE validação final | MSE teste final |
| --- | ---: | ---: | ---: | ---: |
| Linear, 1 → 1 Linear, Min-Max, 200 épocas, taxa 0,05 | 1,84747829 | 4,36295358e-17 | 7,79972252e-17 | 5,28295333e-17 |
| XOR, 2 → 4 Tanh → 1 Sigmoid, sem escala, 1000 épocas, taxa 0,3 | 0,266085698 | 0,000173855266 | 0,000216323341 | 0,000335654753 |

No XOR, as quatro previsões contínuas foram aproximadamente 0,009045,
0,978492, 0,989774 e 0,013000 para 00, 01, 10 e 11. Com limiar 0,5, todas
correspondem à tabela. As cinco repetições de cada padrão tornam possível
a divisão em três conjuntos, mas não constituem validação de generalização
para padrões inéditos. Essa limitação é explicitada no README e no exemplo.

## Pendências e limites

Não foram identificadas pendências nos 12 requisitos auditados. Permanecem
limites documentados do escopo: dados devem estar limpos; treinamento
síncrono; memória/tempo dependem da arquitetura; taxas inadequadas podem
divergir; persistência não inclui histórico, nomes das colunas nem estado do
gerador. A CLI original continua informativa; o Streamlit é a interface
completa da entrega. A API direta `train` é mutável; a proteção de confirmação
em cópia pertence à integração de interface `train_session`.

Não se garante convergência para datasets ou hiperparâmetros arbitrários.
Resultados e igualdade de previsões após JSON foram medidos nas execuções
descritas, não supostos.

## Arquivos desta etapa

**Criados:** `REQUISITOS.md`, `AUDITORIA.md`, `data/linear.csv`, `data/xor.csv`,
`examples/demonstracao.py`, `tests/test_integracao.py`, `tests/test_auditoria.py`.

**Modificados:** `README.md`, `.gitignore`, `src/mlp/neuronio.py`,
`src/mlp/rede.py`, `src/mlp/metricas.py`, `src/mlp/dados.py`,
`src/mlp/treinamento.py`, `src/mlp/interface.py`, `src/mlp/ui/common.py`,
`src/mlp/ui/dataset.py`, `src/mlp/ui/training.py`, `tests/test_app.py`.

Nenhum dataset, teste ou modelo importante foi excluído.
