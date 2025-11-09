import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from queue import PriorityQueue
import time

# --- Configuração Inicial e Utilitários ---
st.set_page_config(layout="wide", page_title="Branch and Bound - Knapsack")

# Variáveis globais para rastreamento (Métricas de Execução)
execution_metrics = {}

# Estrutura para o estado do nó na árvore B&B
Node = namedtuple("Node", ["level", "value", "weight", "bound", "x_vector"])

class BranchAndBoundSolver:
    """
    Implementa o algoritmo Branch and Bound para o Problema da Mochila 0-1 (Knapsack).
    Utiliza uma fila de prioridade para Best-Bound Search (Busca pelo Melhor Limite).
    """

    def __init__(self, projects, capacity):
        """
        Inicializa o solver.
        :param projects: DataFrame com 'Custo (Peso)', 'Retorno (Valor)', e 'Razão V/P'.
        :param capacity: Capacidade máxima da mochila/orçamento.
        """
        self.projects = projects.sort_values(by='Razão V/P', ascending=False).reset_index(drop=True)
        self.W = capacity
        self.n = len(projects)
        self.weights = self.projects['Custo (Peso)'].tolist()
        self.values = self.projects['Retorno (Valor)'].tolist()
        self.best_value = 0
        self.best_x = [0] * self.n
        self.expanded_nodes = 0
        self.feasible_solutions = 0
        self.max_depth = 0
        self.pruned_nodes = 0

    def _calculate_bound(self, node):
        """
        Cálculo do Limite Superior (L_sup) usando Relaxação Linear.
        Permite a adição fracionária de itens não explorados para maximizar o valor
        potencial.
        """
        if node.weight >= self.W:
            return 0  # Nó inviável, limite superior é 0

        # O valor inicial do limite é o valor acumulado até o nó atual
        bound = node.value
        current_weight = node.weight
        j = node.level + 1  # Começa do próximo item a ser considerado

        # Continua adicionando itens fracionariamente (na ordem decrescente V/P)
        while j < self.n and current_weight + self.weights[j] <= self.W:
            current_weight += self.weights[j]
            bound += self.values[j]
            j += 1

        # Adiciona a porção fracionária do último item, se houver espaço
        if j < self.n:
            remaining_weight = self.W - current_weight
            bound += self.values[j] * (remaining_weight / self.weights[j])

        return bound

    def _greedy_solve(self):
        """
        Heurística Gulosa simples para obter um bom valor inicial (Primal Bound).
        Seleciona itens na ordem decrescente da Razão Valor/Peso.
        """
        greedy_value = 0
        greedy_weight = 0
        greedy_x = [0] * self.n
        
        # A lista de projetos já está ordenada por Razão V/P
        for i in range(self.n):
            if greedy_weight + self.weights[i] <= self.W:
                greedy_weight += self.weights[i]
                greedy_value += self.values[i]
                greedy_x[i] = 1
        
        return greedy_value, greedy_x

    def solve(self):
        """
        Executa o algoritmo Branch and Bound.
        """
        start_time = time.time()
        self.expanded_nodes = 0
        self.feasible_solutions = 0
        self.max_depth = 0
        self.pruned_nodes = 0

        # 1. Obter valor inicial (Primal Bound) com Heurística Gulosa
        initial_value, initial_x = self._greedy_solve()
        self.best_value = initial_value
        self.best_x = initial_x
        self.feasible_solutions += 1
        st.info(f"**Primal Bound Inicial (Heurística Gulosa):** {self.best_value:.2f} (Melhor solução conhecida)")

        # 2. Inicializar a Fila de Prioridade (Best-Bound Search)
        # Prioridade é baseada no limite superior (bound), de forma decrescente (por isso o sinal -)
        PQ = PriorityQueue()
        # Nó raiz (Root Node): level=-1, value=0, weight=0, bound calculado
        root_x = [0] * self.n
        root_bound = self._calculate_bound(Node(-1, 0, 0, 0, root_x))
        root_node = Node(-1, 0, 0, root_bound, root_x)
        PQ.put((-root_node.bound, root_node)) # Armazena (-bound, node) para max-heap

        # 3. Processar a árvore B&B
        while not PQ.empty():
            # Pega o nó com o maior bound (maior chance de otimalidade)
            neg_bound, u = PQ.get()
            
            # Se o limite do nó for menor que a melhor solução atual (Primal Bound),
            # pode-se podar o ramo inteiro.
            if u.bound <= self.best_value:
                self.pruned_nodes += 1
                continue # Poda por limite (Bounding)

            # --- Expansão (Branching) ---
            
            # Próximo item/projeto a ser considerado
            i = u.level + 1
            if i >= self.n:
                continue # Fim do ramo

            self.expanded_nodes += 1
            self.max_depth = max(self.max_depth, i)

            # --- Caso 1: Incluir o Item i (x_i = 1) ---
            
            w_included = u.weight + self.weights[i]
            v_included = u.value + self.values[i]
            x_included = u.x_vector[:]
            x_included[i] = 1
            
            if w_included <= self.W: # Poda por Inviabilidade (verificada antes do cálculo do bound)
                v_node_bound = self._calculate_bound(Node(i, v_included, w_included, 0, x_included))
                v_node = Node(i, v_included, w_included, v_node_bound, x_included)
                
                # É uma solução viável (Feasible Solution)
                if i == self.n - 1:
                    self.feasible_solutions += 1
                    # Atualiza a melhor solução (Primal Bound) se for melhor
                    if v_included > self.best_value:
                        self.best_value = v_included
                        self.best_x = x_included
                        
                elif v_node_bound > self.best_value: # Poda por Limite
                    # Se o limite ainda for promissor, adiciona à fila
                    PQ.put((-v_node_bound, v_node))
                else:
                    self.pruned_nodes += 1
            else:
                self.pruned_nodes += 1 # Poda por Inviabilidade (Peso Excedido)


            # --- Caso 2: Excluir o Item i (x_i = 0) ---
            
            w_excluded = u.weight
            v_excluded = u.value
            x_excluded = u.x_vector[:]
            x_excluded[i] = 0
            
            # O bound para o caso de exclusão
            w_node_bound = self._calculate_bound(Node(i, v_excluded, w_excluded, 0, x_excluded))
            w_node = Node(i, v_excluded, w_excluded, w_node_bound, x_excluded)

            if i == self.n - 1:
                self.feasible_solutions += 1
                # Atualiza a melhor solução se for melhor (caso o valor acumulado seja melhor)
                if v_excluded > self.best_value:
                    self.best_value = v_excluded
                    self.best_x = x_excluded
                    
            elif w_node_bound > self.best_value: # Poda por Limite
                # Se o limite ainda for promissor, adiciona à fila
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

# --- Geração e Preparação de Dados (Simulando Dataset Kaggle) ---

def generate_and_prepare_data(num_projects=50):
    """
    Gera um dataset sintético de projetos e prepara para o B&B.
    Simula aquisição, limpeza e EDA.
    """
    np.random.seed(42)  # Reprodutibilidade

    # 1. Calcular Custo/Peso
    costs = np.random.randint(10, 100, num_projects)

    # 2. Calcular Retorno/Valor usando o Custo (para correlação)
    returns = np.round(np.random.normal(loc=1.2, scale=0.3, size=num_projects) * costs, 0) + np.random.randint(-15, 15, num_projects)

    # Simulação de dados
    data = {
        'Nome do Projeto': [f'Proj-{i+1:02d}' for i in range(num_projects)],
        # Custo/Peso: Distribuição mais ou menos uniforme, simulando diferentes necessidades de orçamento
        'Custo (Peso)': costs,
        # Retorno/Valor: Correlacionado com o custo, mas com algum ruído
        'Retorno (Valor)': returns,
        # Uma variável categórica simulando o 'Departamento'
        'Departamento': np.random.choice(['TI', 'Marketing', 'P&D', 'Vendas'], num_projects, p=[0.4, 0.3, 0.1, 0.2]),
        # Simulação de valores faltantes (para a etapa de limpeza)
        'Risco Estimado': np.random.choice([np.nan, 'Baixo', 'Médio', 'Alto'], num_projects, p=[0.1, 0.5, 0.3, 0.1]),
    }
    df = pd.DataFrame(data)
    
    # 1.2 Limpeza e Padronização
    # Remoção de valores faltantes (simplesmente remove para este exemplo)
    df.dropna(subset=['Risco Estimado'], inplace=True) 
    
    # 1.3 Mapeamento para Otimização: Adiciona a Razão V/P, crucial para o B&B
    df['Razão V/P'] = df['Retorno (Valor)'] / df['Custo (Peso)']
    
    # Remove projetos inviáveis (retorno <= 0, peso <= 0) - Simulação de limpeza de inconsistências
    df = df[(df['Retorno (Valor)'] > 0) & (df['Custo (Peso)'] > 0)].reset_index(drop=True)
    
    return df

# --- Dashboards e Front-End com Streamlit ---

def data_exploration_dashboard(df, capacity):
    """Dashboard para Análise Exploratória de Dados (EDA)."""
    st.header("1. Análise Exploratória de Dados (EDA)")
    st.caption(f"Dataset sintético de {len(df)} projetos de investimento, criado para simular dados de otimização de portfólio. A análise visa selecionar o subconjunto de projetos que maximiza o Retorno Total sob a restrição orçamentária.")
    
    st.subheader("1.1. Inspeção Inicial e Estatísticas Descritivas")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Estrutura dos Dados**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("**Estatísticas Chave (Custo e Retorno)**")
        desc_stats = df[['Custo (Peso)', 'Retorno (Valor)', 'Razão V/P']].describe().transpose()
        st.dataframe(desc_stats, use_container_width=True)

    st.markdown(f"""
    <p style='font-size:14px; color: #4F8A10;'>
    **Contexto do Problema:** Tentamos selecionar projetos com **Custo Total $\\le {capacity}$** para **maximizar o Retorno Total**.<br>
    A variável 'Razão V/P' é o principal indicador de eficiência, usado para o cálculo do Limite Superior do B&B.
    </p>
    """, unsafe_allow_html=True)
    
    st.subheader("1.2. Visualizações Exploratórias")

    col3, col4 = st.columns(2)
    
    # Gráfico 1: Scatterplot Custo vs Retorno
    with col3:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='Custo (Peso)', y='Retorno (Valor)', hue='Departamento', size='Retorno (Valor)', data=df, ax=ax1, palette='viridis')
        ax1.axhline(df['Retorno (Valor)'].median(), color='red', linestyle='--', alpha=0.6, label='Mediana Retorno')
        ax1.set_title('Retorno vs Custo por Projeto')
        ax1.legend(title='Departamento', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig1)

    # Gráfico 2: Distribuição da Razão V/P
    with col4:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Razão V/P'], bins=15, kde=True, ax=ax2, color='skyblue')
        ax2.axvline(df['Razão V/P'].mean(), color='darkorange', linestyle='-', label='Média')
        ax2.legend()
        st.pyplot(fig2)

def algorithm_dashboard(df, best_value, best_x, capacity):
    """Dashboard de Resultados e Análise do Algoritmo."""
    
    # 3. Processar Resultados
    best_projects_df = df.copy()
    # Pega o vetor de solução (0 ou 1) e alinha com os projetos ordenados pelo B&B
    # O BranchAndBoundSolver ordena o DF internamente, então precisamos usar a ordem correta
    best_projects_df = best_projects_df.sort_values(by='Razão V/P', ascending=False).reset_index(drop=True)
    best_projects_df['Selecionado'] = best_x
    
    solution_df = best_projects_df[best_projects_df['Selecionado'] == 1]
    total_cost = solution_df['Custo (Peso)'].sum()
    
    st.header("2. Resultados do Branch and Bound e Métricas")

    col1, col2, col3, col4 = st.columns(4)
    
    # Indicadores da Solução Ótima
    col1.metric("Retorno Ótimo (Z)", f"R$ {best_value:,.2f}")
    col2.metric("Custo Total Utilizado", f"R$ {total_cost:,.2f}")
    col3.metric("Capacidade Máxima", f"R$ {capacity:,.2f}")
    col4.metric("Projetos Selecionados", len(solution_df))

    st.subheader("2.1. Métricas de Execução do Algoritmo")
    
    if execution_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tempo Total (s)", f"{execution_metrics['Tempo Total (s)']:.4f}")
        m2.metric("Nós Expandidos", execution_metrics['Nós Expandidos'])
        m3.metric("Nós Podados", execution_metrics['Nós Podados'])
        m4.metric("Soluções Viáveis Encontradas", execution_metrics['Soluções Viáveis'])
        st.write(f"Profundidade Máxima da Árvore: {execution_metrics['Profundidade Máxima']}")

    # CORREÇÃO: Escapar as chaves do LaTeX "$L_{sup}$" para evitar KeyErrors no .format()
    st.markdown("""
    <p style='font-size:14px; color: #155724; background-color: #D4EDDA; border-radius: 5px; padding: 10px;'>
    **Evidência de Poda:** O algoritmo podou 
    **<span style='font-weight:bold;'>{pruned} nós</span>** (ramificações) porque o Limite Superior de Retorno ($L_{{sup}}$) que eles poderiam alcançar era inferior à **Melhor Solução Conhecida (Primal Bound)**, ou porque violaram a restrição de capacidade.
    </p>
    """.format(pruned=execution_metrics.get('Nós Podados', 0)), unsafe_allow_html=True)
    
    st.subheader("2.2. Solução Ótima Encontrada (Projetos Selecionados)")
    
    solution_table = solution_df[['Nome do Projeto', 'Custo (Peso)', 'Retorno (Valor)', 'Razão V/P', 'Departamento']].reset_index(drop=True)
    st.dataframe(solution_table, use_container_width=True)
    
    # 4. Comparação com Heurística Gulosa
    st.subheader("2.3. Comparação de Desempenho (Branch and Bound vs. Gulosa)")
    
    # Calcula a solução Gulosa (usando o método auxiliar)
    solver_temp = BranchAndBoundSolver(df, capacity) # Reutiliza a classe para a gulosa
    greedy_value, _ = solver_temp._greedy_solve()
    
    data_comparison = pd.DataFrame({
        'Método': ['Branch and Bound (Ótimo)', 'Heurística Gulosa'],
        'Retorno Total': [best_value, greedy_value],
        'Diferença (%)': [0, (greedy_value - best_value) / best_value * 100]
    })
    
    st.dataframe(data_comparison, use_container_width=True, hide_index=True,
                 column_config={'Retorno Total': st.column_config.NumberColumn(format="R$ %.2f")})

    # 5. Análise de Sensibilidade (Capacidade)
    st.subheader("2.4. Análise de Sensibilidade - Impacto da Capacidade")
    st.markdown(f"**Parâmetro Variável:** Capacidade Orçamentária ($W$)")

    # Calcula e exibe a solução para uma capacidade menor
    new_capacity = capacity * 0.75
    solver_low = BranchAndBoundSolver(df, new_capacity)
    low_value, _ = solver_low.solve()
    
    st.info(f"""
    **Cenário de Sensibilidade (75% da Capacidade):**
    - Se a Capacidade fosse reduzida para **R$ {new_capacity:.2f}** (75% de R$ {capacity:.2f}), 
    o Retorno Ótimo seria **R$ {low_value:.2f}** (uma queda de R$ {(best_value - low_value):.2f}).
    Isso mostra o impacto direto da restrição de recurso na função objetivo.
    """)
    

# --- Função Principal do Streamlit ---

def main():
    """Centraliza a execução do Streamlit."""
    
    st.title("🚢 Branch and Bound para Otimização de Portfólio de Projetos")
    st.caption("Selecione um número de projetos (os dados do projeto serão selecionados de forma aleatória) e uma capacidade orçamentária. Ao clicar para executar a otimização, nosso algoritmo irá construir o melhor portfólio possível")
    st.markdown("---")
    
    # --- Side Bar: Parâmetros e Configuração ---
    with st.sidebar:
        st.header("Configuração do Problema")
        num_projects = st.slider("Número de Projetos (Itens)", 10, 100, 50)
        
        # O total do peso na simulação de 50 projetos é ~2700. Um bom W é 1/3 disso.
        default_capacity = int(num_projects * 45 / 3) 
        
        capacity = st.number_input("Capacidade Orçamentária (W)", min_value=100, value=default_capacity, step=50)
        
        st.markdown("---")
        st.subheader("Execução do Algoritmo")
        run_button = st.button("Executar Branch and Bound")

    # --- 1. Aquisição e Preparo de Dados ---
    # Gerar e Preparar os dados
    projects_df = generate_and_prepare_data(num_projects)
    
    if 'projects_df' not in st.session_state:
        st.session_state.projects_df = projects_df
        
    # --- Dashboard EDA (Seção 4.2) ---
    data_exploration_dashboard(st.session_state.projects_df, capacity)
    
    st.markdown("---")
    
    if run_button:
        st.subheader("3. Execução do Algoritmo de Otimização (Branch and Bound)")
        st.warning("Executando B&B... Pode levar alguns segundos dependendo do número de projetos.")
        
        # 2. Modelagem e Implementação B&B (Seções 2 e 3)
        solver = BranchAndBoundSolver(st.session_state.projects_df, capacity)
        
        # Executa o solver
        best_value, best_x = solver.solve()
        
        st.success("🎉 Otimização Concluída!")
        
        # 4. Front-End e Dashboards (Seções 4.3 e 4.4)
        algorithm_dashboard(st.session_state.projects_df, best_value, best_x, capacity)
    else:
        st.info("Clique em **Executar Branch and Bound** na barra lateral para iniciar a otimização.")


if __name__ == "__main__":
    main()