# -------------------------------------------------------------------
# app.py
# -------------------------------------------------------------------
# Interface do usuário com Streamlit para o Otimizador de Estoque B&B.
#
# Cobre os itens 4.1, 4.2, 4.3, 4.4, 5.1 e 5.2 do escopo.
#
# Para executar:
# 1. Certifique-se de ter o bnb_solver.py e dados_itens_knapsack.csv na pasta
# 2. No terminal, execute: streamlit run app.py
# -------------------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importar as funções do nosso solver# -------------------------------------------------------------------
# app.py
# -------------------------------------------------------------------
# Interface do usuário com Streamlit para o Otimizador de Estoque B&B.
#
# Cobre os itens 4.1, 4.2, 4.3, 4.4, 5.1 e 5.2 do escopo.
#
# Para executar:
# 1. Certifique-se de ter o bnb_solver.py e dados_itens_knapsack.csv na pasta
# 2. No terminal, execute: streamlit run app.py
# -------------------------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importar as funções do nosso solver
# (Requer que bnb_solver.py esteja na mesma pasta)
try:
    from bnb_solver import solve_branch_and_bound, solve_greedy, Item, calculate_bound
except ImportError:
    st.error("ERRO: O arquivo 'bnb_solver.py' não foi encontrado. Certifique-se de que ele está na mesma pasta que o 'app.py'.")
    st.stop()

# ---
# Configuração da Página
# ---
st.set_page_config(
    page_title="Otimizador de Estoque (B&B)",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Sistema de Otimização de Estoque (Branch and Bound)")
st.write("""
Este sistema utiliza o algoritmo **Branch and Bound** para resolver o **Problema da Mochila 0/1**, 
selecionando o portfólio de itens de estoque que **maximiza o lucro total** sem exceder um **orçamento de capital (W)**.
""")

# ---
# (Item 1.3 / 4.2) Carregamento de Dados (Cache)
# ---
@st.cache_data
def carregar_dados():
    """
    Carrega os dados processados (v, w, eficiencia) do Passo 1.
    """
    try:
        df = pd.read_csv("dados_itens_knapsack.csv")
        # Garantir a ordenação por eficiência, crucial para B&B e Guloso
        df = df.sort_values(by='eficiencia', ascending=False)
        return df
    except FileNotFoundError:
        st.error("ERRO: Arquivo 'dados_itens_knapsack.csv' não encontrado.")
        st.info("Por favor, execute o script do Passo 1 (preparo e EDA) primeiro.")
        st.stop()

df_itens_final = carregar_dados()


# ---
# (Item 4.2) Funções de Plotagem do Dashboard de EDA
# ---
@st.cache_data
def plotar_histogramas(df):
    """Gera os gráficos de histograma para o EDA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(df['v'], bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Distribuição do Lucro Total por Item (v)')
    axes[0].set_xlabel('Lucro (v) - Escala Log')
    axes[0].set_ylabel('Contagem')
    axes[0].set_xscale('log')

    sns.histplot(df['w'], bins=50, kde=True, color='orange', ax=axes[1])
    axes[1].set_title('Distribuição do Custo Total por Item (w)')
    axes[1].set_xlabel('Custo (w) - Escala Log')
    axes[1].set_ylabel('Contagem')
    axes[1].set_xscale('log')
    
    fig.suptitle('EDA: Histogramas de Custo e Lucro', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

@st.cache_data
def plotar_scatter(df):
    """Gera o gráfico de dispersão (Custo x Lucro) para o EDA."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x='w',
        y='v',
        alpha=0.6,
        ax=ax
    )
    ax.set_title('Relação Custo (w) vs. Lucro (v) por Item')
    ax.set_xlabel('Custo Total (w) - Escala Log')
    ax.set_ylabel('Lucro Total (v) - Escala Log')
    ax.set_xscale('log')
    ax.set_yscale('log')
    return fig


# ---
# (Item 4.1) Interface de Usuário (Sidebar de Parâmetros)
# ---
st.sidebar.header("⚙️ Parâmetros da Otimização")

# (Item 5.2) Sensibilidade e Robustez: Orçamento (W)
w_max = int(df_itens_final['w'].sum())
w_default = int(w_max * 0.1) # Sugerir 10% do custo total como default

W_budget = st.sidebar.slider(
    "Orçamento de Estoque (W)",
    min_value=10000,
    max_value=w_max,
    value=w_default,
    step=1000,
    format="R$ %d",
    key="slider_orcamento_1"
)

# (Requisito 2 de Qualidade) Ajuste do Tamanho do Problema (N)
N_items = st.sidebar.slider(
    "Nº de Itens para Otimizar (Top N por eficiência)",
    min_value=10,
    max_value=min(1000, len(df_itens_final)), # Limitar a 1000 ou total
    value=100, # Default (bom para demonstração)
    step=10
)

st.sidebar.info(f"""
**Informações do Problema:**
* **Total de Itens (Base):** {len(df_itens_final)}
* **Itens a Otimizar (N):** {N_items}
* **Orçamento (W):** R$ {W_budget:,.2f}
""")

run_button = st.sidebar.button("🚀 Executar Otimização B&B")


# ---
# (Item 4.2, 4.3, 4.4) Dashboards Principais (Tabs)
# ---
tab1, tab2, tab3 = st.tabs([
    "📊 Análise Exploratória (EDA)",
    "⚙️ Execução do Algoritmo B&B",
    "🏆 Resultados da Otimização"
])


# ---
# TAB 1: Dashboard de Análise de Dados (EDA)
# ---
with tab1:
    st.header("📊 Análise Exploratória dos Dados dos Itens")
    st.write(f"Análise baseada no dataset completo de **{len(df_itens_final)}** itens únicos processados.")
    
    st.subheader("Amostra dos Dados (Itens formatados para Mochila 0/1)")
    st.dataframe(df_itens_final.head(10))

    st.subheader("Estatísticas Descritivas (v, w, eficiencia)")
    st.dataframe(df_itens_final[['v', 'w', 'eficiencia']].describe())
    
    st.subheader("Visualização da Distribuição de Custo e Lucro")
    st.pyplot(plotar_histogramas(df_itens_final))
    
    st.subheader("Visualização da Relação Custo vs. Lucro")
    st.pyplot(plotar_scatter(df_itens_final))

# ---
# TAB 2: Dashboard do Algoritmo B&B (Execução)
# ---
with tab2:
    st.header("⚙️ Execução e Métricas do Branch and Bound")
    
    if not run_button:
        st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Otimização'.")
    
    if run_button:
        # 1. Preparar os dados para o solver (Top N itens)
        st.write(f"Iniciando otimização com **N={N_items}** itens e **W={W_budget:,.2f}**...")
        
        df_problema = df_itens_final.head(N_items)
        
        # Converter DataFrame para a lista de namedtuple 'Item'
        items_list = [
            Item(index=row['StockCode'], v=row['v'], w=row['w'], eficiencia=row['eficiencia'])
            for index, row in df_problema.iterrows()
        ]

        # 2. Executar a Heurística Gulosa (Baseline - Item 5.1)
        with st.spinner("Executando Heurística Gulosa (Baseline)..."):
            (
                greedy_profit, 
                greedy_weight, 
                greedy_indices
            ) = solve_greedy(W_budget, items_list)
        
        st.session_state['greedy_results'] = {
            'profit': greedy_profit,
            'weight': greedy_weight,
            'indices': greedy_indices
        }
        
        # 3. Executar o Branch and Bound
        with st.spinner(f"Executando Branch and Bound... Isso pode levar alguns segundos."):
            (
                bnb_profit, 
                bnb_weight, 
                bnb_indices, 
                bnb_metrics
            ) = solve_branch_and_bound(W_budget, items_list)

        st.success("Otimização B&B Concluída!")
        
        # Guardar resultados no st.session_state para usar na Tab 3
        st.session_state['bnb_results'] = {
            'profit': bnb_profit,
            'weight': bnb_weight,
            'indices': bnb_indices,
            'metrics': bnb_metrics
        }
        st.session_state['df_problema'] = df_problema
        st.session_state['results_available'] = True
        st.session_state['W_budget'] = W_budget

        # 4. (Item 4.3) Exibir Métricas e Evidências de Poda
        st.subheader("Métricas de Execução do B&B (Item 3.2)")
        metrics = bnb_metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Tempo de Execução", f"{metrics['execution_time']:.4f} s")
        col2.metric("Nós Expandidos", f"{metrics['nodes_expanded']:,}")
        col3.metric("Profundidade Máxima", f"{metrics['max_depth_reached']:,}")
        
        st.subheader("Evidências de Poda (Item 2.3)")
        col4, col5, col6 = st.columns(3)
        col4.metric("Soluções Viáveis Encontradas", f"{metrics['solutions_found']:,}")
        col5.metric("Podas por Limite (Bound)", f"{metrics['pruning_by_bound']:,}")
        col6.metric("Podas por Inviabilidade", f"{metrics['pruning_by_infeasibility']:,}")

# ---
# TAB 3: Dashboard de Resultados da Otimização
# ---
with tab3:
    st.header("🏆 Resultados da Otimização")

    if 'results_available' not in st.session_state:
        st.info("Execute o algoritmo na aba 'Execução do Algoritmo B&B' para ver os resultados.")
    else:
        # Carregar resultados salvos
        bnb_results = st.session_state['bnb_results']
        greedy_results = st.session_state['greedy_results']
        df_problema = st.session_state['df_problema']
        W_budget = st.session_state['W_budget']
        
        # (Item 4.4) Solução Final e Função Objetivo
        st.subheader("Solução Ótima (Branch and Bound)")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Lucro Máximo (Função Objetivo)", 
            f"R$ {bnb_results['profit']:,.2f}"
        )
        col2.metric(
            "Custo Total (Orçamento Usado)",
            f"R$ {bnb_results['weight']:,.2f}"
        )
        col3.metric(
            "Itens Selecionados",
            f"{len(bnb_results['indices'])}"
        )
        
        # (Item 5.1) Comparação com Heurística
        st.subheader("Comparação com Heurística Gulosa")
        
        delta_profit = bnb_results['profit'] - greedy_results['profit']
        
        col4, col5, col6 = st.columns(3)
        col4.metric(
            "Lucro (Guloso)", 
            f"R$ {greedy_results['profit']:,.2f}"
        )
        col5.metric(
            "Custo (Guloso)",
            f"R$ {greedy_results['weight']:,.2f}"
        )
        col6.metric(
            "Melhoria (B&B vs. Guloso)",
            f"R$ {delta_profit:,.2f}",
            help="Mostra o quanto o B&B foi melhor que a heurística simples."
        )

        # (Item 4.4) Visualização Contextual (Tabela de Itens)
        st.subheader("Itens Selecionados para Estocar (Solução Ótima)")
        
        if len(bnb_results['indices']) > 0:
            # Filtrar o dataframe original para mostrar os itens selecionados
            df_solucao_otima = df_problema[
                df_problema['StockCode'].isin(bnb_results['indices'])
            ]
            st.dataframe(df_solucao_otima[[
                'StockCode', 'Description', 'v', 'w', 'eficiencia', 'Quantity', 'UnitPrice'
            ]])
        else:
            st.warning("Nenhum item foi selecionado com os parâmetros atuais.")
# (Requer que bnb_solver.py esteja na mesma pasta)
try:
    from bnb_solver import solve_branch_and_bound, solve_greedy, Item, calculate_bound
except ImportError:
    st.error("ERRO: O arquivo 'bnb_solver.py' não foi encontrado. Certifique-se de que ele está na mesma pasta que o 'app.py'.")
    st.stop()

# ---
# Configuração da Página
# ---
st.set_page_config(
    page_title="Otimizador de Estoque (B&B)",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Sistema de Otimização de Estoque (Branch and Bound)")
st.write("""
Este sistema utiliza o algoritmo **Branch and Bound** para resolver o **Problema da Mochila 0/1**, 
selecionando o portfólio de itens de estoque que **maximiza o lucro total** sem exceder um **orçamento de capital (W)**.
""")

# ---
# (Item 1.3 / 4.2) Carregamento de Dados (Cache)
# ---
@st.cache_data
def carregar_dados():
    """
    Carrega os dados processados (v, w, eficiencia) do Passo 1.
    """
    try:
        df = pd.read_csv("dados_itens_knapsack.csv")
        # Garantir a ordenação por eficiência, crucial para B&B e Guloso
        df = df.sort_values(by='eficiencia', ascending=False)
        return df
    except FileNotFoundError:
        st.error("ERRO: Arquivo 'dados_itens_knapsack.csv' não encontrado.")
        st.info("Por favor, execute o script do Passo 1 (preparo e EDA) primeiro.")
        st.stop()

df_itens_final = carregar_dados()


# ---
# (Item 4.2) Funções de Plotagem do Dashboard de EDA
# ---
@st.cache_data
def plotar_histogramas(df):
    """Gera os gráficos de histograma para o EDA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(df['v'], bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Distribuição do Lucro Total por Item (v)')
    axes[0].set_xlabel('Lucro (v) - Escala Log')
    axes[0].set_ylabel('Contagem')
    axes[0].set_xscale('log')

    sns.histplot(df['w'], bins=50, kde=True, color='orange', ax=axes[1])
    axes[1].set_title('Distribuição do Custo Total por Item (w)')
    axes[1].set_xlabel('Custo (w) - Escala Log')
    axes[1].set_ylabel('Contagem')
    axes[1].set_xscale('log')
    
    fig.suptitle('EDA: Histogramas de Custo e Lucro', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

@st.cache_data
def plotar_scatter(df):
    """Gera o gráfico de dispersão (Custo x Lucro) para o EDA."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x='w',
        y='v',
        alpha=0.6,
        ax=ax
    )
    ax.set_title('Relação Custo (w) vs. Lucro (v) por Item')
    ax.set_xlabel('Custo Total (w) - Escala Log')
    ax.set_ylabel('Lucro Total (v) - Escala Log')
    ax.set_xscale('log')
    ax.set_yscale('log')
    return fig


# ---
# (Item 4.1) Interface de Usuário (Sidebar de Parâmetros)
# ---
st.sidebar.header("⚙️ Parâmetros da Otimização")

# (Item 5.2) Sensibilidade e Robustez: Orçamento (W)
w_max = int(df_itens_final['w'].sum())
w_default = int(w_max * 0.1) # Sugerir 10% do custo total como default

W_budget = st.sidebar.slider(
    "Orçamento de Estoque (W)",
    min_value=10000,
    max_value=w_max,
    value=w_default,
    step=1000,
    format="R$ %d"
)

# (Requisito 2 de Qualidade) Ajuste do Tamanho do Problema (N)
N_items = st.sidebar.slider(
    "Nº de Itens para Otimizar (Top N por eficiência)",
    min_value=10,
    max_value=min(1000, len(df_itens_final)), # Limitar a 1000 ou total
    value=100, # Default (bom para demonstração)
    step=10
)

st.sidebar.info(f"""
**Informações do Problema:**
* **Total de Itens (Base):** {len(df_itens_final)}
* **Itens a Otimizar (N):** {N_items}
* **Orçamento (W):** R$ {W_budget:,.2f}
""")

run_button = st.sidebar.button("🚀 Executar Otimização B&B")


# ---
# (Item 4.2, 4.3, 4.4) Dashboards Principais (Tabs)
# ---
tab1, tab2, tab3 = st.tabs([
    "📊 Análise Exploratória (EDA)",
    "⚙️ Execução do Algoritmo B&B",
    "🏆 Resultados da Otimização"
])


# ---
# TAB 1: Dashboard de Análise de Dados (EDA)
# ---
with tab1:
    st.header("📊 Análise Exploratória dos Dados dos Itens")
    st.write(f"Análise baseada no dataset completo de **{len(df_itens_final)}** itens únicos processados.")
    
    st.subheader("Amostra dos Dados (Itens formatados para Mochila 0/1)")
    st.dataframe(df_itens_final.head(10))

    st.subheader("Estatísticas Descritivas (v, w, eficiencia)")
    st.dataframe(df_itens_final[['v', 'w', 'eficiencia']].describe())
    
    st.subheader("Visualização da Distribuição de Custo e Lucro")
    st.pyplot(plotar_histogramas(df_itens_final))
    
    st.subheader("Visualização da Relação Custo vs. Lucro")
    st.pyplot(plotar_scatter(df_itens_final))

# ---
# TAB 2: Dashboard do Algoritmo B&B (Execução)
# ---
with tab2:
    st.header("⚙️ Execução e Métricas do Branch and Bound")
    
    if not run_button:
        st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Otimização'.")
    
    if run_button:
        # 1. Preparar os dados para o solver (Top N itens)
        st.write(f"Iniciando otimização com **N={N_items}** itens e **W={W_budget:,.2f}**...")
        
        df_problema = df_itens_final.head(N_items)
        
        # Converter DataFrame para a lista de namedtuple 'Item'
        items_list = [
            Item(index=row['StockCode'], v=row['v'], w=row['w'], eficiencia=row['eficiencia'])
            for index, row in df_problema.iterrows()
        ]

        # 2. Executar a Heurística Gulosa (Baseline - Item 5.1)
        with st.spinner("Executando Heurística Gulosa (Baseline)..."):
            (
                greedy_profit, 
                greedy_weight, 
                greedy_indices
            ) = solve_greedy(W_budget, items_list)
        
        st.session_state['greedy_results'] = {
            'profit': greedy_profit,
            'weight': greedy_weight,
            'indices': greedy_indices
        }
        
        # 3. Executar o Branch and Bound
        with st.spinner(f"Executando Branch and Bound... Isso pode levar alguns segundos."):
            (
                bnb_profit, 
                bnb_weight, 
                bnb_indices, 
                bnb_metrics
            ) = solve_branch_and_bound(W_budget, items_list)

        st.success("Otimização B&B Concluída!")
        
        # Guardar resultados no st.session_state para usar na Tab 3
        st.session_state['bnb_results'] = {
            'profit': bnb_profit,
            'weight': bnb_weight,
            'indices': bnb_indices,
            'metrics': bnb_metrics
        }
        st.session_state['df_problema'] = df_problema
        st.session_state['results_available'] = True
        st.session_state['W_budget'] = W_budget

        # 4. (Item 4.3) Exibir Métricas e Evidências de Poda
        st.subheader("Métricas de Execução do B&B (Item 3.2)")
        metrics = bnb_metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Tempo de Execução", f"{metrics['execution_time']:.4f} s")
        col2.metric("Nós Expandidos", f"{metrics['nodes_expanded']:,}")
        col3.metric("Profundidade Máxima", f"{metrics['max_depth_reached']:,}")
        
        st.subheader("Evidências de Poda (Item 2.3)")
        col4, col5, col6 = st.columns(3)
        col4.metric("Soluções Viáveis Encontradas", f"{metrics['solutions_found']:,}")
        col5.metric("Podas por Limite (Bound)", f"{metrics['pruning_by_bound']:,}")
        col6.metric("Podas por Inviabilidade", f"{metrics['pruning_by_infeasibility']:,}")

# ---
# TAB 3: Dashboard de Resultados da Otimização
# ---
with tab3:
    st.header("🏆 Resultados da Otimização")

    if 'results_available' not in st.session_state:
        st.info("Execute o algoritmo na aba 'Execução do Algoritmo B&B' para ver os resultados.")
    else:
        # Carregar resultados salvos
        bnb_results = st.session_state['bnb_results']
        greedy_results = st.session_state['greedy_results']
        df_problema = st.session_state['df_problema']
        W_budget = st.session_state['W_budget']
        
        # (Item 4.4) Solução Final e Função Objetivo
        st.subheader("Solução Ótima (Branch and Bound)")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Lucro Máximo (Função Objetivo)", 
            f"R$ {bnb_results['profit']:,.2f}"
        )
        col2.metric(
            "Custo Total (Orçamento Usado)",
            f"R$ {bnb_results['weight']:,.2f}"
        )
        col3.metric(
            "Itens Selecionados",
            f"{len(bnb_results['indices'])}"
        )
        
        # (Item 5.1) Comparação com Heurística
        st.subheader("Comparação com Heurística Gulosa")
        
        delta_profit = bnb_results['profit'] - greedy_results['profit']
        
        col4, col5, col6 = st.columns(3)
        col4.metric(
            "Lucro (Guloso)", 
            f"R$ {greedy_results['profit']:,.2f}"
        )
        col5.metric(
            "Custo (Guloso)",
            f"R$ {greedy_results['weight']:,.2f}"
        )
        col6.metric(
            "Melhoria (B&B vs. Guloso)",
            f"R$ {delta_profit:,.2f}",
            help="Mostra o quanto o B&B foi melhor que a heurística simples."
        )

        # (Item 4.4) Visualização Contextual (Tabela de Itens)
        st.subheader("Itens Selecionados para Estocar (Solução Ótima)")
        
        if len(bnb_results['indices']) > 0:
            # Filtrar o dataframe original para mostrar os itens selecionados
            df_solucao_otima = df_problema[
                df_problema['StockCode'].isin(bnb_results['indices'])
            ]
            st.dataframe(df_solucao_otima[[
                'StockCode', 'Description', 'v', 'w', 'eficiencia', 'Quantity', 'UnitPrice'
            ]])
        else:
            st.warning("Nenhum item foi selecionado com os parâmetros atuais.")