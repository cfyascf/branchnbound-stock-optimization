# -------------------------------------------------------------------
# bnb_solver.py
# -------------------------------------------------------------------
# Contém a implementação do algoritmo Branch and Bound
# para o Problema da Mochila 0/1 (Gestão de Estoque).
#
# Cobre os itens 2, 3 e 5.1 do escopo.
# -------------------------------------------------------------------

import pandas as pd
import time
from collections import namedtuple

# (Item 2.1) Definição Formal do Modelo
#
# Problema: Problema da Mochila 0/1 (Knapsack 0/1)
#
#   Indices:
#     i = 1, ..., N (itens de estoque, StockCode)
#
#   Parâmetros:
#     v_i = Lucro total esperado do item i (calculado no Passo 1)
#     w_i = Custo total de estoque do item i (calculado no Passo 1)
#     W   = Orçamento total de capital (capacidade da mochila)
#
#   Variáveis de Decisão:
#     x_i = 1 se o item i é selecionado, 0 caso contrário
#
#   Função Objetivo (Maximizar Lucro):
#     Maximizar Z = Σ (v_i * x_i)
#
#   Restrição (Orçamento):
#     Σ (w_i * x_i) <= W

# Usamos namedtuple para clareza ao acessar os itens
Item = namedtuple("Item", ['index', 'v', 'w', 'eficiencia'])


# ---
# (Item 2.2) Hipótese de Relaxação (Cálculo do Bound)
# ---

def calculate_bound(node_level, node_profit, node_weight, W, items):
    """
    Calcula o Limite Superior (Upper Bound) para um nó na árvore de busca
    usando a relaxação linear (mochila fracionária).
    """
    if node_weight > W:
        return 0  # Nó inviável, bound é 0

    n = len(items)
    bound_profit = node_profit
    bound_weight = node_weight
    
    # Itera pelos itens restantes (do nível atual em diante)
    for i in range(node_level, n):
        item = items[i]
        
        # Se o item inteiro cabe, adiciona-o
        if bound_weight + item.w <= W:
            bound_weight += item.w
            bound_profit += item.v
        else:
            # Se não cabe, adiciona a fração que couber
            # Esta é a "relaxação linear"
            espaco_restante = W - bound_weight
            fracao_lucro = item.eficiencia * espaco_restante
            
            bound_profit += fracao_lucro
            bound_weight += espaco_restante # Agora bound_weight == W
            break # A mochila está cheia
            
    return bound_profit


# ---
# (Item 5.1) Comparação de Desempenho (Heurística Gulosa)
# ---

def solve_greedy(W, items):
    """
    Resolve o problema usando uma heurística gulosa simples (baseline).
    Ordena por eficiência e pega os itens que cabem.
    """
    # Itens já vêm ordenados por eficiência
    n = len(items)
    total_profit = 0
    total_weight = 0
    selected_items_indices = []

    for i in range(n):
        item = items[i]
        if total_weight + item.w <= W:
            total_weight += item.w
            total_profit += item.v
            selected_items_indices.append(item.index)
    
    return total_profit, total_weight, selected_items_indices


# ---
# (Item 3.1, 2.3, 3.2) Implementação do Branch and Bound (B&B)
# ---

def solve_branch_and_bound(W, items):
    """
    Resolve o Problema da Mochila 0/1 usando Branch and Bound.
    Usa uma estratégia de busca em profundidade (DFS) com pilha (stack).
    """
    n = len(items)
    
    # ----------------------------------------------------
    # (Item 3.2) Métricas de Execução
    # ----------------------------------------------------
    start_time = time.time()
    metrics = {
        'nodes_expanded': 0,
        'pruning_by_infeasibility': 0,
        'pruning_by_bound': 0,
        'solutions_found': 0,
        'max_depth_reached': 0,
        'execution_time': 0,
    }

    # (Item 2.3) Política de Busca: Pilha (Stack) para Busca em Profundidade (DFS)
    # Nó da pilha: (level, profit, weight)
    # level: índice do item a ser decidido
    # profit: lucro acumulado até este nó
    # weight: peso (custo) acumulado até este nó
    stack = []
    
    # Nó raiz (nível 0, lucro 0, peso 0)
    root_node = (0, 0, 0)
    stack.append(root_node)
    
    # (Item 2.3) Condição de Parada
    # O loop para quando a pilha fica vazia.
    
    # Limite Inferior (Lower Bound - LB): o melhor lucro inteiro encontrado
    max_profit = 0.0
    best_solution_indices = [] # Para rastrear os itens da melhor solução

    # Usamos uma pilha auxiliar para rastrear o caminho (itens incluídos)
    # Formato do nó no 'path_stack': ( (level, profit, weight), item_index )
    # Se item_index for -1, significa que o item do nível anterior não foi incluído.
    path_stack = [ (root_node, -1) ]
    
    
    # Inicia a busca
    while stack:
        
        # 1. Pega um nó da pilha (DFS)
        current_level, current_profit, current_weight = stack.pop()
        
        # Recupera o caminho que levou a este nó
        current_path_node, item_idx_incluido = path_stack.pop()
        
        # Monta a solução parcial atual
        current_solution_indices = [item_idx for (node, item_idx) in path_stack if item_idx != -1]
        if item_idx_incluido != -1:
            current_solution_indices.append(item_idx_incluido)

        metrics['nodes_expanded'] += 1
        metrics['max_depth_reached'] = max(metrics['max_depth_reached'], current_level)

        # 2. Verifica se é uma solução completa (folha da árvore)
        if current_level == n:
            # Se for uma folha, seu 'current_profit' é um resultado final
            if current_profit > max_profit:
                max_profit = current_profit
                best_solution_indices = current_solution_indices.copy()
                metrics['solutions_found'] += 1
            continue # Não há mais filhos para expandir


        # 3. BRANCHING (Gerar filhos)
        
        item = items[current_level]
        
        # ---
        # Filho 1: INCLUINDO o item (Ramo do "Sim")
        # ---
        weight_com_item = current_weight + item.w
        profit_com_item = current_profit + item.v
        
        # (Poda por Inviabilidade)
        if weight_com_item <= W:
            
            # (Poda por Otimização)
            # Encontramos uma nova solução viável (pode não ser completa ainda)
            # Atualiza o LB (max_profit) se for a melhor até agora
            if profit_com_item > max_profit:
                max_profit = profit_com_item
                metrics['solutions_found'] += 1
                # Atualiza a melhor solução
                best_solution_indices = current_solution_indices.copy()
                best_solution_indices.append(item.index)
            
            # (Poda por Limite)
            # Calcula o bound para este filho
            bound_com_item = calculate_bound(
                current_level + 1, profit_com_item, weight_com_item, W, items
            )
            
            if bound_com_item > max_profit:
                # Se o limite é promissor, adiciona à pilha
                new_node = (current_level + 1, profit_com_item, weight_com_item)
                stack.append(new_node)
                path_stack.append((new_node, item.index)) # Rastreia inclusão
            else:
                metrics['pruning_by_bound'] += 1 # PODA
                
        else:
            metrics['pruning_by_infeasibility'] += 1 # PODA
            

        # ---
        # Filho 2: NÃO INCLUINDO o item (Ramo do "Não")
        # ---
        weight_sem_item = current_weight
        profit_sem_item = current_profit
        
        # (Poda por Limite)
        # Calcula o bound para este filho
        bound_sem_item = calculate_bound(
            current_level + 1, profit_sem_item, weight_sem_item, W, items
        )
        
        if bound_sem_item > max_profit:
            # Se o limite é promissor, adiciona à pilha
            new_node = (current_level + 1, profit_sem_item, weight_sem_item)
            stack.append(new_node)
            path_stack.append((new_node, -1)) # Rastreia não-inclusão
        else:
            metrics['pruning_by_bound'] += 1 # PODA

    # 4. Fim do loop
    metrics['execution_time'] = time.time() - start_time
    
    # Calcula o peso final da melhor solução
    final_weight = sum(item.w for item in items if item.index in best_solution_indices)

    return max_profit, final_weight, best_solution_indices, metrics


# ---
# (Item 5.3) Testes e Execução Principal (para validar)
# ---

if __name__ == "__main__":
    
    print("--- Iniciando Teste do Solver B&B (Passos 2, 3, 5) ---")

    # (Item 3.3) Reprodutibilidade: Carrega os dados gerados no Passo 1
    try:
        df_itens_final = pd.read_csv("dados_itens_knapsack.csv")
    except FileNotFoundError:
        print("Erro: Arquivo 'dados_itens_knapsack.csv' não encontrado.")
        print("Execute o script do Passo 1 primeiro.")
        exit()

    # (Item 3.3) Requisito de Qualidade: Ajustar o tamanho do problema
    # O dataset completo (3k+ itens) demora muito.
    # Vamos pegar os N itens mais eficientes, conforme definido no Passo 1.
    
    N_ITEMS_PARA_OTIMIZAR = 100 # Ajuste este valor para testar desempenho
    W_ORCAMENTO = 500000.0      # Orçamento (Capacidade da Mochila)
    
    # Garante que os dados estão ordenados pela 'eficiencia' (crucial!)
    df_itens_final = df_itens_final.sort_values(by='eficiencia', ascending=False)
    
    # Seleciona os N primeiros itens
    df_problema = df_itens_final.head(N_ITEMS_PARA_OTIMIZAR)

    # Converte o DataFrame para a estrutura de 'Item' (namedtuple)
    # Usamos o 'StockCode' como 'index'
    items_list = [
        Item(index=row['StockCode'], v=row['v'], w=row['w'], eficiencia=row['eficiencia'])
        for index, row in df_problema.iterrows()
    ]

    print(f"\nProblema Configurado:")
    print(f"  Orçamento (W): {W_ORCAMENTO:,.2f}")
    print(f"  Nº de Itens (N): {len(items_list)} (Top {N_ITEMS_PARA_OTIMIZAR} por eficiência)")

    # 1. Executando a Heurística Gulosa (Baseline)
    print("\nExecutando Heurística Gulosa (Item 5.1)...")
    greedy_profit, greedy_weight, _ = solve_greedy(W_ORCAMENTO, items_list)
    print(f"  Resultado Guloso (Lucro): {greedy_profit:,.2f}")
    print(f"  Resultado Guloso (Custo): {greedy_weight:,.2f}")


    # 2. Executando o Branch and Bound (Solução Ótima)
    print("\nExecutando Branch and Bound (Item 3.1)...")
    (
        bnb_profit, 
        bnb_weight, 
        bnb_indices, 
        bnb_metrics
    ) = solve_branch_and_bound(W_ORCAMENTO, items_list)
    
    print("\n--- RESULTADOS DA OTIMIZAÇÃO B&B ---")
    
    print(f"\nLucro Ótimo Encontrado: {bnb_profit:,.2f}")
    print(f"Custo Total da Solução: {bnb_weight:,.2f} (de {W_ORCAMENTO:,.2f})")
    print(f"Número de Itens Selecionados: {len(bnb_indices)}")
    
    print("\n--- MÉTRICAS DE EXECUÇÃO (Item 3.2) ---")
    print(f"  Tempo de Execução: {bnb_metrics['execution_time']:.4f} segundos")
    print(f"  Nós Expandidos: {bnb_metrics['nodes_expanded']:,}")
    print(f"  Profundidade Máxima: {bnb_metrics['max_depth_reached']:,}")
    print(f"  Soluções Viáveis Encontradas: {bnb_metrics['solutions_found']:,}")
    
    print("\n--- EVIDÊNCIAS DE PODA (Item 2.3) ---")
    print(f"  Podas por Inviabilidade: {bnb_metrics['pruning_by_infeasibility']:,}")
    print(f"  Podas por Limite (Bound): {bnb_metrics['pruning_by_bound']:,}")
    
    print("\n--- FIM DO PASSO 2 e 3 ---")