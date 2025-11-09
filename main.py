import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from queue import PriorityQueue
import time

# --- Configuração Inicial e Utilitários ---
st.set_page_config(layout="wide", page_title="Branch and Bound - Otimização de Estoque")

# Variáveis globais para rastreamento
execution_metrics = {}

# Estrutura para o estado do nó na árvore B&B
Node = namedtuple("Node", ["level", "value", "weight", "bound", "x_vector"])

class BranchAndBoundSolver:
    """
    Implementa o algoritmo Branch and Bound para o Problema da Mochila 0-1.
    (Adaptado para Otimização de Estoque).
    """

    def __init__(self, items, capacity):
        """Inicializa o solver."""
        # A ordenação é feita pela Razão Lucro/Volume (L/V)
        self.items = items.sort_values(by='Razão L/V', ascending=False).reset_index(drop=True)
        self.W = capacity
        self.n = len(items)
        # weight = Volume (m³), value = Lucro Estimado
        self.weights = self.items['Volume (m³)'].tolist() 
        self.values = self.items['Lucro Estimado'].tolist()
        self.best_value = 0
        self.best_x = [0] * self.n
        self.expanded_nodes = 0
        self.feasible_solutions = 0
        self.max_depth = 0
        self.pruned_nodes = 0

    def _calculate_bound(self, node):
        """Cálculo do Limite Superior (L_sup) usando Relaxação Linear."""
        if node.weight >= self.W:
            return 0  # Nó inviável

        # O valor inicial do limite é o lucro acumulado
        bound = node.value
        current_weight = node.weight
        j = node.level + 1  # Começa do próximo item

        # Continua adicionando itens fracionariamente
        while j < self.n and current_weight + self.weights[j] <= self.W:
            current_weight += self.weights[j]
            bound += self.values[j]
            j += 1

        # Adiciona a porção fracionária
        if j < self.n:
            remaining_weight = self.W - current_weight
            bound += self.values[j] * (remaining_weight / self.weights[j])

        return bound

    def _greedy_solve(self):
        """Heurística Gulosa para obter um Primal Bound inicial."""
        greedy_value = 0
        greedy_weight = 0
        greedy_x = [0] * self.n
        
        # Seleciona itens na ordem L/V
        for i in range(self.n):
            if greedy_weight + self.weights[i] <= self.W:
                greedy_weight += self.weights[i]
                greedy_value += self.values[i]
                greedy_x[i] = 1
        
        return greedy_value, greedy_x

    def solve(self):
        """Executa o algoritmo Branch and Bound."""
        start_time = time.time()
        self.expanded_nodes = 0
        self.feasible_solutions = 0
        self.max_depth = 0
        self.pruned_nodes = 0

        # 1. Obter valor inicial (Primal Bound)
        initial_value, initial_x = self._greedy_solve()
        self.best_value = initial_value
        self.best_x = initial_x
        self.feasible_solutions += 1


        # 2. Inicializar a Fila de Prioridade (Best-Bound Search)
        PQ = PriorityQueue()
        # Nó raiz
        root_x = [0] * self.n
        root_bound = self._calculate_bound(Node(-1, 0, 0, 0, root_x))
        root_node = Node(-1, 0, 0, root_bound, root_x)
        PQ.put((-root_node.bound, root_node)) # max-heap

        # 3. Processar a árvore B&B
        while not PQ.empty():
            neg_bound, u = PQ.get()
            
            # Poda por Limite (Bounding)
            if u.bound <= self.best_value:
                self.pruned_nodes += 1
                continue 

            # --- Expansão (Branching) ---
            
            i = u.level + 1
            if i >= self.n:
                continue 

            self.expanded_nodes += 1
            self.max_depth = max(self.max_depth, i)

            # --- Caso 1: Incluir o Item i (x_i = 1) ---
            
            w_included = u.weight + self.weights[i]
            v_included = u.value + self.values[i]
            x_included = u.x_vector[:]
            x_included[i] = 1
            
            if w_included <= self.W: # Poda por Inviabilidade verificada
                v_node_bound = self._calculate_bound(Node(i, v_included, w_included, 0, x_included))
                v_node = Node(i, v_included, w_included, v_node_bound, x_included)
                
                # É uma solução viável
                if i == self.n - 1:
                    self.feasible_solutions += 1
                    if v_included > self.best_value:
                        self.best_value = v_included
                        self.best_x = x_included
                        
                elif v_node_bound > self.best_value: # Poda por Limite
                    PQ.put((-v_node_bound, v_node))
                else:
                    self.pruned_nodes += 1
            else:
                self.pruned_nodes += 1 # Poda por Inviabilidade (Volume Excedido)


            # --- Caso 2: Excluir o Item i (x_i = 0) ---
            
            w_excluded = u.weight
            v_excluded = u.value
            x_excluded = u.x_vector[:]
            x_excluded[i] = 0
            
            w_node_bound = self._calculate_bound(Node(i, v_excluded, w_excluded, 0, x_excluded))
            w_node = Node(i, v_excluded, w_excluded, w_node_bound, x_excluded)

            if i == self.n - 1:
                self.feasible_solutions += 1
                if v_excluded > self.best_value:
                    self.best_value = v_excluded
                    self.best_x = x_excluded
                    
            elif w_node_bound > self.best_value: # Poda por Limite
                PQ.put((-w_node_bound, w_node))
            else:
                self.pruned_nodes += 1
        
        end_time = time.time()
        
        # Armazenar métricas para o dashboard
        global execution_metrics
        execution_metrics['Tempo Total (s)'] = end_time - start_time
        execution_metrics['Nós Expandidos'] = self.expanded_nodes
        execution_metrics['Nós Podados'] = self.pruned_nodes
        execution_metrics['Soluções Viáveis'] = self.feasible_solutions
        execution_metrics['Profundidade Máxima'] = self.max_depth

        return self.best_value, self.best_x

# --- Geração e Preparação de Dados (Simulando Dataset de Estoque) ---

def generate_and_prepare_data(num_items=50):
    """Gera um dataset sintético de SKUs para otimização de estoque."""
    np.random.seed(42)  # Reprodutibilidade

    # 1. Calcular Volume (m³) - Variável de Restrição (Peso)
    volumes = np.round(np.random.uniform(0.1, 5.0, num_items), 2)

    # 2. Calcular Lucro Estimado - Variável Objetivo (Valor)
    profits = np.round(np.random.normal(loc=20, scale=8, size=num_items) * volumes, 2) + np.random.randint(-10, 50, num_items)

    # Simulação de dados
    data = {
        'Nome do Item (SKU)': [f'SKU-{i+1:03d}' for i in range(num_items)],
        'Volume (m³)': volumes,
        'Lucro Estimado': profits,
        # Uma variável categórica
        'Categoria': np.random.choice(['Eletrônicos', 'Alimentos Secos', 'Limpeza', 'Vestuário'], num_items, p=[0.2, 0.4, 0.1, 0.3]),
        # Simulação de valores faltantes
        'Giro de Estoque': np.random.choice([np.nan, 'Alto', 'Médio', 'Baixo'], num_items, p=[0.05, 0.4, 0.3, 0.25]),
    }
    df = pd.DataFrame(data)
    
    # 1.2 Limpeza e Padronização
    df.dropna(subset=['Giro de Estoque'], inplace=True) 
    
    # 1.3 Mapeamento para Otimização: Adiciona a Razão Lucro/Volume
    df['Razão L/V'] = df['Lucro Estimado'] / df['Volume (m³)']
    
    # Remove itens inviáveis
    df = df[(df['Lucro Estimado'] > 0) & (df['Volume (m³)'] > 0)].reset_index(drop=True)
    
    return df

# --- Dashboards e Front-End com Streamlit ---

def data_exploration_dashboard(df, capacity):
    """Dashboard para Análise Exploratória de Dados (EDA)."""
    st.header("1. Análise Exploratória de Dados (EDA) - Estoque")
    
    st.markdown(f"""
        Análise exploratória do dataset de {len(df)} SKUs (Stock Keeping Units). Esta etapa visa compreender a eficiência espacial (Razão Lucro/Volume) e a qualidade dos dados antes de aplicar o algoritmo Branch and Bound para selecionar o portfólio de itens mais lucrativo.
    """, unsafe_allow_html=True)
    st.subheader("1.1. Inspeção Inicial e Estatísticas Descritivas")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Estrutura dos Dados (SKUs)**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("**Estatísticas Chave (Volume e Lucro)**")
        desc_stats = df[['Volume (m³)', 'Lucro Estimado', 'Razão L/V']].describe().transpose()
        st.dataframe(desc_stats, use_container_width=True)

    st.write("")
    st.write("**Interpretação das Estatísticas Descritivas**")
    st.markdown("""
        <li style='margin-bottom: 8px;'>
            <strong style='color: #007bff;'>count/mean:</strong> O número de itens e a média de Volume, Lucro e Razão L/V. A média da Razão L/V indica a eficiência típica dos SKUs.
        </li>
        <li style='margin-bottom: 8px;'>
            <strong style='color: #007bff;'>std:</strong> O desvio padrão, que mede a dispersão. Um std alto para 'Lucro Estimado' indica grande variação no lucro potencial dos itens.
        </li>
        <li style='margin-bottom: 8px;'>
            <strong style='color: #007bff;'>min/max:</strong> Os valores mínimo e máximo (outliers ou extremos). Eles definem o intervalo de Volume e Lucro que o algoritmo de otimização precisa gerenciar.
        </li>
        <li style='margin-bottom: 8px;'>
            <strong style='color: #007bff;'>25%, 50% (Mediana), 75%:</strong> Os quartis. A Mediana (50%) é o valor central. Comparar a Média com a Mediana ajuda a identificar se a distribuição dos dados é assimétrica (skewed).
        </li>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("**Modelagem**")
    st.markdown(f"""
        **Objetivo do Modelo:** A otimização busca **Maximizar o Lucro Estimado Total** dos SKUs escolhidos.<br>
        **Restrição de Estoque:** O **Volume Total** ocupado deve ser **menor ou igual a $W$ m³** (Capacidade do Armazém).<br>
        **Eficiência:** A **Razão Lucro/Volume** é a chave para identificar quais itens oferecem a melhor **eficiência espacial**.
    """, unsafe_allow_html=True)
    
    st.subheader("1.2. Visualizações Exploratórias")

    col3, col4 = st.columns(2)
    
    # Gráfico 1: Scatterplot Lucro vs Volume
    with col3:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='Volume (m³)', y='Lucro Estimado', hue='Categoria', size='Lucro Estimado', data=df, ax=ax1, palette='Spectral')
        ax1.axhline(df['Lucro Estimado'].median(), color='red', linestyle='--', alpha=0.6, label='Mediana Lucro')
        ax1.set_title('Lucro Estimado vs Volume por SKU')
        ax1.legend(title='Categoria', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig1)
        st.markdown(
            """
            **Interpretação do Gráfico de Dispersão:**
            Este gráfico mostra a relação direta entre o **Volume** (restrição) e o **Lucro** (objetivo).
            SKUs com **baixo Volume e alto Lucro** são os mais *eficientes* e, portanto, os candidatos ideais para o Branch and Bound.
            A cor e o tamanho dos pontos indicam categorias e magnitude do lucro.
            """
        )

    # Gráfico 2: Distribuição da Razão Lucro/Volume
    with col4:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Razão L/V'], bins=15, kde=True, ax=ax2, color='darkgreen')
        ax2.axvline(df['Razão L/V'].mean(), color='orange', linestyle='-', label='Média')
        ax2.legend()
        st.pyplot(fig2)
        st.markdown(
            """
            **Interpretação do Histograma da Razão L/V (Eficiência Espacial):**
            A **Razão L/V** é a métrica central para o Branch and Bound (usada no *bound*).
            Este histograma mostra a distribuição da eficiência. SKUs com **Razão L/V alta** são a prioridade da **Heurística Gulosa** e a base da poda do B&B.
            Uma distribuição mais concentrada (baixo desvio padrão) indica menor variação de eficiência entre os SKUs.
            """
        )

def algorithm_dashboard(df, best_value, best_x, capacity):
    """Dashboard de Resultados e Análise do Algoritmo."""
    
    # 3. Processar Resultados
    best_items_df = df.copy()
    # Usa a ordem correta
    best_items_df = best_items_df.sort_values(by='Razão L/V', ascending=False).reset_index(drop=True)
    best_items_df['Selecionado'] = best_x
    
    solution_df = best_items_df[best_items_df['Selecionado'] == 1]
    total_volume = solution_df['Volume (m³)'].sum()
    
    st.header("2. Resultados do Branch and Bound e Métricas")

    col1, col2, col3, col4 = st.columns(4)
    
    # Indicadores da Solução Ótima
    col1.metric("Lucro Ótimo (Z)", f"R$ {best_value:,.2f}")
    col2.metric("Volume Total Utilizado", f"{total_volume:,.2f} m³")
    col3.metric("Capacidade Máxima Armazenamento", f"{capacity:,.2f} m³")
    col4.metric("SKUs Selecionados", len(solution_df))

    st.subheader("2.1. Métricas de Execução do Algoritmo")
    
    if execution_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tempo Total (s)", f"{execution_metrics['Tempo Total (s)']:.4f}")
        m2.metric("Nós Expandidos", execution_metrics['Nós Expandidos'])
        m3.metric("Nós Podados", execution_metrics['Nós Podados'])
        m4.metric("Soluções Viáveis Encontradas", execution_metrics['Soluções Viáveis'])
        st.write(f"Profundidade Máxima da Árvore: {execution_metrics['Profundidade Máxima']}")

    # Evidência de Poda
    st.markdown("""
    <p style='font-size:14px; color: #155724; background-color: #D4EDDA; border-radius: 5px; padding: 10px;'>
    **Evidência de Poda:** O algoritmo podou 
    **<span style='font-weight:bold;'>{pruned} nós</span>** (ramificações) porque o Limite Superior de Lucro ($L_{{sup}}$) que eles poderiam alcançar era inferior à **Melhor Solução Conhecida (Primal Bound)**, ou porque violaram a restrição de capacidade.
    </p>
    """.format(pruned=execution_metrics.get('Nós Podados', 0)), unsafe_allow_html=True)
    
    st.subheader("2.2. Solução Ótima Encontrada (SKUs Selecionados para Estoque)")
    
    solution_table = solution_df[['Nome do Item (SKU)', 'Volume (m³)', 'Lucro Estimado', 'Razão L/V', 'Categoria']].reset_index(drop=True)
    st.dataframe(solution_table, use_container_width=True)
    
    # 4. Comparação com Heurística Gulosa
    st.subheader("2.3. Comparação de Desempenho (Branch and Bound vs. Gulosa)")
    
    # Calcula a solução Gulosa
    solver_temp = BranchAndBoundSolver(df, capacity) # Reutiliza a classe
    greedy_value, _ = solver_temp._greedy_solve()
    
    data_comparison = pd.DataFrame({
        'Método': ['Branch and Bound (Ótimo)', 'Heurística Gulosa (Razão L/V)'],
        'Lucro Total': [best_value, greedy_value],
        'Diferença (%)': [0, (greedy_value - best_value) / best_value * 100]
    })
    
    st.dataframe(data_comparison, use_container_width=True, hide_index=True,
                 column_config={'Lucro Total': st.column_config.NumberColumn(format="R$ %.2f")})

    # 5. Análise de Sensibilidade (Capacidade)
    st.subheader("2.4. Análise de Sensibilidade - Impacto da Capacidade de Volume")

    # Calcula e exibe a solução para uma capacidade menor
    new_capacity = capacity * 0.75
    solver_low = BranchAndBoundSolver(df, new_capacity)
    low_value, _ = solver_low.solve()
    
    st.info(f"""
    **Cenário de Sensibilidade (75% da Capacidade):**
    - Se a Capacidade Volumétrica fosse reduzida para **{new_capacity:.2f} m³** (75% de {capacity:.2f} m³), 
    o Lucro Ótimo seria {low_value:.2f} (uma queda de R$ {(best_value - low_value):.2f}).
    Isso mostra o impacto direto da restrição de espaço na maximização do lucro.
    """)

    st.write("Esta análise é crucial para avaliar a **robustez** da nossa solução ótima e entender o **custo de oportunidade** do espaço no armazém. Ao variar a capacidade ($W$), medimos o impacto direto no Lucro Máximo. Isso permite ao gerente de estoque justificar decisões de expansão ou planejar cenários de restrição de espaço com base no retorno financeiro.")
    

# --- Função Principal do Streamlit ---

def main():
    """Centraliza a execução do Streamlit."""
    
    st.title("Otimização de Estoque (Branch and Bound)")
    st.markdown("### Seleção de unidades para máximo lucro sob restrição de volume")
    st.caption("Selecione o número de itens possíveis (os dados dos itens serão selecionados de forma randomica) e a capacidade do armazém. Ao executar a otimização, nosso algoritmo irá encontrar a melhor combinação de itens pra se ter em estoque para maximizar o lucro.")
    st.markdown("---")
    
    # --- Side Bar: Parâmetros e Configuração ---
    with st.sidebar:
        st.header("Configuração do Problema")
        num_items = st.slider("Número de SKUs (Itens)", 10, 100, 50)
        
        # Define capacidade default
        default_capacity = round(num_items * 2.8 / 3) 
        
        capacity = st.number_input("Capacidade Máxima de Volume (m³)", min_value=10.0, value=float(default_capacity), step=1.0)
        
        st.markdown("---")
        st.subheader("Execução do Algoritmo")
        run_button = st.button("Executar Branch and Bound")

    # --- 1. Aquisição e Preparo de Dados ---
    items_df = generate_and_prepare_data(num_items)
    
    if 'items_df' not in st.session_state:
        st.session_state.items_df = items_df
        
    # --- Dashboard EDA (Seção 4.2) ---
    data_exploration_dashboard(st.session_state.items_df, capacity)
    
    st.markdown("---")
    
    if run_button:
        st.subheader("3. Execução do Algoritmo de Otimização (Branch and Bound)")
        st.warning("Executando B&B... Pode levar alguns segundos dependendo do número de SKUs.")
        
        # 2. Modelagem e Implementação B&B 
        solver = BranchAndBoundSolver(st.session_state.items_df, capacity)
        
        # Executa o solver
        best_value, best_x = solver.solve()
        
        st.success("🎉 Otimização Concluída!")
        
        # 4. Front-End e Dashboards
        algorithm_dashboard(st.session_state.items_df, best_value, best_x, capacity)
    else:
        st.info("Clique em **Executar Branch and Bound** na barra lateral para iniciar a otimização.")


if __name__ == "__main__":
    main()