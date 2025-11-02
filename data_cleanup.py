# -------------------------------------------------------------------
# PASSO 1: AQUISIÇÃO, PREPARO E ANÁLISE DE DADOS
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configuração inicial
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ---
# 1.1. Aquisição de Dados (Dataset do Kaggle)
# ---
print("Iniciando 1.1: Aquisição de Dados...")

# URL do dataset (UCI E-Commerce Data)
# Link Kaggle: https://www.kaggle.com/datasets/carrie1/ecommerce-data
# Fonte Original: http://archive.ics.uci.edu/ml/datasets/Online+Retail
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# Tentar carregar os dados. Este dataset é conhecido por problemas de encoding
try:
    df = pd.read_excel(data_url)
    print("Dados carregados com sucesso (Excel).")
except Exception as e:
    print(f"Erro ao carregar Excel: {e}. Tentando CSV (se disponível)...")
    # Se você baixou o CSV do Kaggle, use:
    # df = pd.read_csv("data.csv", encoding="ISO-8859-1")

print("\nEstrutura inicial dos dados:")
df.info()

# ---
# 1.2. Limpeza e Padronização
# ---
print("\nIniciando 1.2: Limpeza e Padronização...")

# Remover registros sem 'CustomerID' ou 'Description'
# Um item sem descrição não pode ser estocado
df.dropna(subset=['CustomerID', 'Description'], inplace=True)

# Remover transações com quantidade negativa (são devoluções/cancelamentos)
df_limpo = df[df['Quantity'] > 0].copy()

# Remover transações com preço unitário zero
df_limpo = df_limpo[df_limpo['UnitPrice'] > 0]

# Converter InvoiceDate para datetime (para futuras análises, se necessário)
df_limpo['InvoiceDate'] = pd.to_datetime(df_limpo['InvoiceDate'])

# Vamos focar em um único país para este problema, ex: 'United Kingdom'
df_limpo = df_limpo[df_limpo['Country'] == 'United Kingdom']

print(f"Registros restantes após limpeza: {len(df_limpo)}")

# ---
# 1.3. Mapeamento para um Problema de Otimização (Problema da Mochila 0/1)
# ---
print("\nIniciando 1.3: Mapeamento para Problema da Mochila...")

# DECISÃO DE MODELAGEM:
# Para transformar transações em "itens" para estocar, vamos:
# 1. Definir o custo e o lucro. Assumiremos uma margem de lucro fixa.
#    Esta é uma simplificação que DEVE ser documentada no seu relatório.
MARGEM_LUCRO = 0.40  # 40% de margem
TAXA_CUSTO = 1.0 - MARGEM_LUCRO # 60% de custo

df_limpo['Custo_Unitario'] = df_limpo['UnitPrice'] * TAXA_CUSTO
df_limpo['Lucro_Unitario'] = df_limpo['UnitPrice'] * MARGEM_LUCRO

# 2. Agregar os dados por 'StockCode' (nosso "item")
# Queremos saber a demanda histórica total (soma da Quantidade)
# e o lucro/custo médio por item.

agg_funcs = {
    'Description': 'first',      # Pegar a primeira descrição encontrada
    'Quantity': 'sum',           # Demanda total histórica
    'Custo_Unitario': 'mean',    # Custo unitário médio
    'Lucro_Unitario': 'mean',    # Lucro unitário médio
    'UnitPrice': 'mean'          # Preço unitário médio
}

df_itens = df_limpo.groupby('StockCode').agg(agg_funcs).reset_index()

# 3. Calcular 'v_i' (Valor/Lucro) e 'w_i' (Peso/Custo) para cada item
# v_i = Lucro total esperado se estocarmos este item (baseado na demanda histórica)
# w_i = Custo total para estocar este item (baseado na demanda histórica)

df_itens['v (Lucro Total)'] = df_itens['Lucro_Unitario'] * df_itens['Quantity']
df_itens['w (Custo Total)'] = df_itens['Custo_Unitario'] * df_itens['Quantity']

# 4. Limpeza final dos itens
# Remover itens que não geram lucro ou não têm custo (provavelmente erros de dados)
df_itens = df_itens[(df_itens['v (Lucro Total)'] > 0) & (df_itens['w (Custo Total)'] > 0)]

# 5. Calcular 'Eficiência' (v_i / w_i)
# Isso é crucial para a heurística e para o cálculo do bound
df_itens['eficiencia'] = df_itens['v (Lucro Total)'] / df_itens['w (Custo Total)']

# Renomear colunas para o algoritmo
df_itens.rename(columns={
    'v (Lucro Total)': 'v',
    'w (Custo Total)': 'w'
}, inplace=True)

# Ordenar por eficiência (útil para o B&B)
df_itens_final = df_itens.sort_values(by='eficiencia', ascending=False)

print(f"Mapeamento concluído. Número de itens únicos para otimização: {len(df_itens_final)}")
print("\nAmostra da base de itens pronta para o B&B:")
print(df_itens_final[['StockCode', 'Description', 'v', 'w', 'eficiencia']].head())


# ---
# 1.4. Análise Exploratória de Dados (EDA)
# ---
print("\nIniciando 1.4: Análise Exploratória (EDA)...")

# Opcional: Salvar os dados processados para usar no Streamlit
df_itens_final.to_csv("dados_itens_knapsack.csv", index=False)
print("Dados dos itens salvos em 'dados_itens_knapsack.csv'")

# 1. Estatísticas Descritivas
print("\nEstatísticas Descritivas (v, w):")
print(df_itens_final[['v', 'w', 'eficiencia']].describe())

# 2. Visualização Exploratória (Histogramas)
# A distribuição é muito assimétrica (poucos itens muito caros/lucrativos)
# Usaremos escala de log para melhor visualização

plt.figure(figsize=(14, 6))

# Histograma de 'v' (Lucro Total)
plt.subplot(1, 2, 1)
sns.histplot(df_itens_final['v'], bins=50, kde=True)
plt.title('Distribuição do Lucro Total por Item (v)')
plt.xlabel('Lucro (v) - Escala Log')
plt.ylabel('Contagem')
plt.xscale('log') # Escala de log

# Histograma de 'w' (Custo Total)
plt.subplot(1, 2, 2)
sns.histplot(df_itens_final['w'], bins=50, kde=True, color='orange')
plt.title('Distribuição do Custo Total por Item (w)')
plt.xlabel('Custo (w) - Escala Log')
plt.ylabel('Contagem')
plt.xscale('log') # Escala de log

plt.suptitle('EDA: Histogramas de Custo e Lucro', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("eda_histogramas.png")
print("Salvo 'eda_histogramas.png'")

# 3. Visualização Exploratória (Scatterplot Custo x Lucro)
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_itens_final,
    x='w',
    y='v',
    alpha=0.6,
    # Opcional: colorir pela eficiência
    # hue='eficiencia', 
    # palette='viridis'
)
plt.title('Relação Custo (w) vs. Lucro (v) por Item')
plt.xlabel('Custo Total (w) - Escala Log')
plt.ylabel('Lucro Total (v) - Escala Log')
plt.xscale('log')
plt.yscale('log')

# Interpretação (para seu relatório):
# O gráfico mostra que nem sempre o item mais caro é o mais lucrativo.
# Existem muitos itens de baixo custo e baixo lucro (canto inferior esquerdo).
# Nosso desafio é encontrar a "fronteira" de itens (canto superior esquerdo)
# que maximiza o lucro (eixo Y) sem estourar o orçamento (eixo X).

plt.savefig("eda_scatter_custo_lucro.png")
print("Salvo 'eda_scatter_custo_lucro.png'")

print("\n--- FIM DO PASSO 1 ---")