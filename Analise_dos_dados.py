"""1º Importar as bibliotecas necessárias"""
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

"""2° Carregar os dados da tabela de registro de  compras e geolocalização dos clientes passadas dos clientes"""
caminho_clientes = "./Banco de Dados/clientes.csv"
caminho_vendas = './Banco de Dados/vendas_de_produtos.csv'
dados_clientes = pd.read_csv(caminho_clientes)
dados_vendas = pd.read_csv(caminho_vendas)

""" Dados da Tabela 
● ID_Cliente: identificação única do cliente;
● Data: quando a compra foi realizada;
● ID_Produto: identificação única do produto comprado;
● Descrição_Produto: descrição detalhada do produto;
● Quantidade: total de unidades do produto vendidas;
● Preço_Unitário: preço de venda de uma unidade do produto;
● ID_Pedido: identificação única do pedido de venda;
● Desconto: desconto no pedido de venda como um todo;
● Frete: frete pago no pedido de venda como um todo
● Total_do_Pedido: valor total do pedido de venda.
"""

"""2.1ºResolver valores vazio (colunas e linhas que todos ou alguns valores são vazios, estes serão excluidas)"""
dados_clientes = dados_clientes.dropna(how="all", axis=1)  # Exlui colunas e/ou linhas que são completamentes vazias
dados_clientes = dados_clientes.dropna(how="any", axis=0)  # Se tiver um codigo vazio, a linha e/ou coluna sera excluida

dados_vendas = dados_vendas.dropna(how="all", axis=1)  # Exlui colunas e/ou linhas que são completamentes vazias
dados_vendas = dados_vendas.dropna(how="any", axis=0)  # Se tiver um codigo vazio, a linha e/ou coluna sera excluida


""" 3° Fazer a junção das tabelas com base em uma coluna em comum para melhor análise e desenvolvimento"""
# Por exemplo, se as tabelas tiverem uma coluna chamada 'id' em comum
tabela_junta = pd.merge(dados_clientes, dados_vendas, on='ID_Cliente')
# Selecionar as colunas desejadas para a terceira tabela
colunas_selecionadas = ['ID_Cliente', 'Quantidade', 'Data', 'Frete', 'Desconto', 'Total_do_Pedido']
terceira_tabela = tabela_junta[colunas_selecionadas]
# Ordenar a tabela por frequência de valores na coluna 'quantidade'
terceira_tabela = terceira_tabela.sort_values(by='Quantidade', ascending=False)
# Salvar a terceira tabela num arquivo CSV
terceira_tabela.to_csv('dados_analisados.csv', index=False)

""" 4° Calcular a  probabilidade em que os clientes vão recomprar,
    usando como ferramenta a  regressão linear e arvore de decisão almejando um acerto acima de 95%"""

tabela = pd.read_csv('dados_analisados.csv')
# Calcular a quantidade de dias desde uma data de referência
data_atual = datetime.now()
data_duas_semanas_atras = data_atual - timedelta(weeks=14)

# Converter a coluna "Data" para o formato de data
tabela['Data'] = pd.to_datetime(tabela['Data'])
tabela['Data'] = (pd.Timestamp('now') - tabela['Data']).dt.days
# Exibir o DataFrame com a coluna de dias
print(tabela)
# Gráfico de calor para pré processamento
sns.heatmap(tabela.corr(), cmap='Blues', annot=True)
plt.show()

# Criando a IA
x = tabela[['ID_Cliente', 'Data', 'Frete', 'Desconto', 'Quantidade']]
y = tabela['Total_do_Pedido']
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# Treina IA
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# Previsão
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
print(f' o R² da Regressão Linear é de :{r2_score(y_teste, previsao_regressaolinear)*100:.2f}%')
print(f'O R² da Arvore de decisão é de : {r2_score(y_teste, previsao_arvoredecisao)*100:.2f}%')

# Visualização Gráfica das previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsão Árvore Decisão"] = previsao_arvoredecisao
tabela_auxiliar["Previsão Regressão Linear"] = previsao_regressaolinear
df_resultado = tabela_auxiliar.reset_index(drop=True)
fig = plt.figure(figsize=(15, 5))
sns.lineplot(data=tabela_auxiliar)
print(tabela_auxiliar)
plt.show()

tabela_teste = x_teste.copy()

# Previsão com Árvore de Decisão, pois teve um acerto aproximado de 97%
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# Criar uma coluna na tabela com base no limiar de probabilidade (0.2)
tabela_teste['Compra_Prevista_ArvoreDecisao'] = previsao_arvoredecisao
tabela_teste['Compra_Prevista_ArvoreDecisao'] = tabela_teste['Compra_Prevista_ArvoreDecisao'] > 0.2

# Separar as previsões de compras verdadeiras (True) e falsas (False)
compras_true_arvoredecisao = tabela_teste[tabela_teste['Compra_Prevista_ArvoreDecisao'] == True]
compras_false_arvoredecisao = tabela_teste[tabela_teste['Compra_Prevista_ArvoreDecisao'] == False]

# Calcular as porcentagens para Árvore de Decisão
total_compras_arvoredecisao = len(compras_true_arvoredecisao) + len(compras_false_arvoredecisao)
porcentagem_compras_arvoredecisao = len(compras_true_arvoredecisao) / total_compras_arvoredecisao * 100
porcentagem_nao_compras_arvoredecisao = len(compras_false_arvoredecisao) / total_compras_arvoredecisao * 100

# Exibir as porcentagens
print("Porcentagem de clientes que compraram (Árvore de Decisão): {:.2f}%".format
      (porcentagem_compras_arvoredecisao))
print("Porcentagem de clientes que não compraram (Árvore de Decisão): {:.2f}%".format
      (porcentagem_nao_compras_arvoredecisao))

# Salvar os resultados em arquivos CSV
compras_true_arvoredecisao.to_csv('Clientes_com_recompra.csv', index=False)
compras_false_arvoredecisao.to_csv('Clientes_com_possivel_não_comprarem.csv', index=False)
