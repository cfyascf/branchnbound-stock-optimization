# 📦 Sistema Branch and Bound para Otimização de Estoque  

Este projeto implementa o algoritmo **Branch and Bound (B&B)** para resolver o **Problema da Mochila 0-1**, aplicado ao cenário de **Gestão de Estoques e Seleção de SKUs**.  

O objetivo é **maximizar o lucro total**, selecionando um conjunto de itens (SKUs) para armazenar, respeitando uma **restrição de volume máximo disponível no armazém**.  

O sistema foi desenvolvido em **Python**, utilizando o **Streamlit** para criar uma **interface interativa** com dashboards e visualizações.  

---

## 🧩 1. Contexto do Problema e Dados  

### 🔹 1.1 Seleção e Mapeamento do Dataset  

**Problema Modelado:** Problema da Mochila 0-1 (Knapsack)  
**Cenário:** Seleção de quais SKUs (Itens em Estoque) devem ser mantidos, dada a limitação de espaço físico no armazém.  

**Link de Referência (Conceitual):**  
O dataset é gerado **sinteticamente** com base em dados típicos de Inventário/Vendas (simulando, por exemplo, um dataset de vendas de supermercado ou e-commerce).  

**Variáveis Relevantes (Adaptadas):**  
| Variável | Descrição | Símbolo |
|-----------|------------|----------|
| Nome do Item (SKU) | Identificador do produto | - |
| Volume (m³) | Espaço ocupado por unidade (Restrição/Peso) | $w_i$ |
| Lucro Estimado | Valor de lucro potencial do item | $v_i$ |
| Razão Lucro/Volume | Indicador de eficiência espacial | $v_i / w_i$ |

---

### 🔹 1.2 Modelagem Formal do Problema  

O objetivo é **maximizar o Lucro Estimado total dos itens selecionados**, respeitando a restrição de **volume máximo de armazenamento ($W$)**.  

**Variáveis de Decisão ($x_i$):**  

$$
x_i \in \{0, 1\} \quad \forall i \in \text{SKUs}
$$

Onde:  
- $x_i = 1$ → item *i* é selecionado para o estoque  
- $x_i = 0$ → item *i* é descartado  

**Função Objetivo (Maximização do Lucro):**  

$$
\text{Maximizar } Z = \sum_i v_i x_i
$$

**Restrição (Capacidade Volumétrica):**  

$$
\sum_i w_i x_i \le W
$$

---

## 🧮 2. Implementação do Branch and Bound  

### 🔸 2.1 Estratégia de Busca e Estrutura  

**Política de Busca:**  
Busca pelo **Melhor Limite (Best-Bound Search)** — utiliza uma **Fila de Prioridade** para explorar o nó com o maior limite superior ($L_{sup}$).  

**Estrutura de Estado (Nó):**  
Cada nó na árvore é definido por:  
- **Nível:** índice do item sendo considerado  
- **Valor Atual:** lucro total acumulado  
- **Peso Atual:** volume total acumulado  
- **Limite Superior (Bound):** melhor lucro possível, considerando frações dos itens restantes  

---

### 🔸 2.2 Hipótese de Relaxação (Cálculo do Bound)  

O **Limite Superior ($L_{sup}$)** é calculado usando a **Relaxação Linear da Mochila**,  
adicionando itens **fracionariamente** na ordem **decrescente da Razão Lucro/Volume ($v_i/w_i$)**  
até atingir a capacidade máxima $W$.  

---

### 🔸 2.3 Critérios de Poda (Pruning) e Parada  

- **Poda por Inviabilidade:** se o Volume Atual exceder $W$  
- **Poda por Limite (Bounding):** se o Limite Superior ($L_{sup}$) ≤ Melhor Lucro já encontrado (Primal Bound)  
- **Condição de Parada:** quando a Fila de Prioridade estiver vazia  

---

## 🚀 3. Execução  

Para executar o sistema localmente, siga os passos abaixo:  

### 1️⃣ Instalar dependências  
```bash
pip install -r requirements.txt