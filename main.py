import tkinter as tk # interface
import numpy as np
from tkinter import simpledialog # interface
import networkx as nx #exibir grafo
import matplotlib.pyplot as plt
import heapq #prim
from collections import deque # busca em largura
from queue import PriorityQueue
import json

class Grafo:
  cont_aresta = 0
  # construtor da classe função chamada quando instancia o objeto dessa classe
  def __init__(self, vertices, direcionado, ponderado):
    self.vertices = np.array(vertices)
    self.n_vertices = len(vertices)
    self.arestas = {}
    self.direcionado = direcionado
    self.ponderado = ponderado
    self.matriz = np.zeros(shape = (self.n_vertices,self.n_vertices))

  def adicionar_vertice(self, id = ''):
    self.matriz = np.vstack([self.matriz, np.zeros(self.n_vertices)]) #adiciona linha 'vstack'
    self.n_vertices = self.n_vertices + 1
    self.matriz = np.hstack([self.matriz, np.zeros((self.n_vertices,1))])  #adiciona coluna 'hstack'
    self.vertices = np.append(self.vertices,id)

  def adicionar_aresta(self, v1, v2, peso = 1, id = ''):
    if self.direcionado == False:
      self.matriz[v1][v2] = peso
      self.matriz[v2][v1] = peso
      self.cont_aresta = self.cont_aresta + 1
    else:
      self.matriz[v1][v2] = peso
      self.cont_aresta = self.cont_aresta + 1
    id = id if id != '' else self.cont_aresta
    self.arestas[id] = {'v1':v1, 'v2':v2}

  def remover_vertice(self,vertice):
    self.matriz = np.delete(self.matriz,vertice,0)
    self.matriz = np.delete(self.matriz,vertice,1)
    self.vertices = np.delete(self.vertices, vertice)

  def remover_aresta(self, v1, v2):
    if self.direcionado == False:
      self.matriz[v1][v2] = 0
      self.matriz[v2][v1] = 0
    else:
      self.matriz[v1][v2] = 0


def exibe_grafo(grafo, colors=None, caminho=None): #M2 considerar cor (vetor de cor) e caminho(sequencia de vertices)
  plt.clf()
  if (grafo.direcionado):
    GD = nx.DiGraph()
    GD.add_nodes_from(grafo.vertices)
    for i in range(len(grafo.matriz)):
      for j in range(len(grafo.matriz)):
        if (grafo.matriz[i][j] != 0):
          V1 = grafo.vertices[i]
          V2 = grafo.vertices[j]
          GD.add_edge(V1, V2)
          GD[V1][V2]['peso'] = grafo.matriz[i][j]

    pesos = nx.get_edge_attributes(GD, 'peso')
    posicoes = nx.circular_layout(GD)
    if colors is not None: #vertice coloridos
        nx.draw_networkx_nodes(GD, posicoes, node_color=colors, node_size=500)
        nx.draw_networkx_labels(GD, posicoes, font_color='black', font_size=12)
        nx.draw_networkx_edges(GD, posicoes, edge_color='black', arrows=True)
        nx.draw_networkx_edge_labels(GD, pos=posicoes, edge_labels=pesos)
    else: #vertice padrão
        nx.draw_circular(GD, with_labels=True)
        _ = nx.draw_networkx_edge_labels(GD, pos=posicoes, edge_labels=pesos)
        if caminho is not None: #aresta
            caminho = [(grafo.vertices[caminho[i]], grafo.vertices[caminho[i + 1]]) for i in range(len(caminho) - 1)]
            cores = ['red' if (u, v) in caminho or (v, u) in caminho else 'black' for (u, v) in GD.edges]
            nx.draw_networkx_edges(GD, posicoes, edge_color=cores)
  else:
    GN = nx.Graph()
    GN.add_nodes_from(grafo.vertices)
    for i in range(len(grafo.matriz)):
      for j in range(len(grafo.matriz)):
        if (grafo.matriz[i][j] != 0):
          V1 = grafo.vertices[i]
          V2 = grafo.vertices[j]
          GN.add_edge(V1, V2)
          GN[V1][V2]['peso'] = grafo.matriz[i][j]

    pesos = nx.get_edge_attributes(GN, 'peso')
    posicoes = nx.circular_layout(GN)

    if colors is not None:
        nx.draw_networkx_nodes(GN, posicoes, node_color=colors, node_size=500)
        nx.draw_networkx_labels(GN, posicoes, font_color='black', font_size=12)
        nx.draw_networkx_edges(GN, posicoes, edge_color='black', arrows=True)
    else:
        nx.draw_circular(GN, with_labels=True)
        nx.draw_networkx_edge_labels(GN, pos=posicoes, edge_labels=pesos)
        if caminho is not None:
            #converte a lista de vertice para lista de arestas
            caminho = [(grafo.vertices[caminho[i]], grafo.vertices[caminho[i + 1]]) for i in range(len(caminho) - 1)]
            #percorre todas as arestas do grafo se a aresta está na lista de caminhos pinta de vermelho se não preto
            cores = ['red' if (u, v) in caminho or (v, u) in caminho else 'black' for (u, v) in GN.edges]
            nx.draw_networkx_edges(GN, posicoes, edge_color=cores)
  plt.show() #cria a janela do matplot para exibir
def prim(adj_matrix):
    num_vertices = adj_matrix.shape[0]  # número de vértices do grafo
    visited = set()  # conjunto de vértices visitados
    min_span_tree = []  # lista de tuplas representando as arestas da árvore geradora mínima
    start_vertex = np.random.randint(num_vertices)  # escolha aleatória do vértice inicial
    visited.add(start_vertex)
    edges = []
    for u in range(num_vertices):
        if u != start_vertex:
            weight = adj_matrix[start_vertex, u]
            if weight != 0:
                edges.append((weight, start_vertex, u))  # lista de arestas incidentes em start_vertex
    heapq.heapify(edges)  # transforma a lista de arestas em uma heap

    while edges:
        weight, u, v = heapq.heappop(edges)  # seleciona a aresta de menor peso
        if v not in visited:  # se o vértice v ainda não foi visitado
            visited.add(v)
            min_span_tree.append((u, v, weight))
            for u2 in range(num_vertices):
                if u2 != v and adj_matrix[v, u2] != 0:  # lista de arestas incidentes em v
                    w = adj_matrix[v, u2]
                    if u2 not in visited:
                        heapq.heappush(edges, (w, v, u2))  # adiciona as arestas incidentes em v não visitadas na heap
    return min_span_tree

def bfs(adj_matrix, start_vertex):
    # Inicializa a lista de vértices visitados e a fila de vértices a visitar
    visited = []
    queue = deque([start_vertex])

    # Enquanto ainda houver vértices na fila de vértices a visitar
    while queue:
        # Pega o próximo vértice da fila de vértices a visitar
        current_vertex = queue.popleft()

        # Se o vértice ainda não foi visitado, adiciona à lista de visitados
        if current_vertex not in visited:
            visited.append(current_vertex)

            # Adiciona os vértices adjacentes ao vértice atual na fila de vértices a visitar
            for i in range(len(adj_matrix)):
                if adj_matrix[current_vertex][i] != 0 and i not in visited:
                    queue.append(i)

    return visited

def dfs(adj_matrix, start_vertex):
    # Inicializa a lista de vértices visitados e a pilha de vértices a visitar
    visited = []
    stack = [start_vertex]

    # Enquanto ainda houver vértices na pilha de vértices a visitar
    while stack:
        # Pega o próximo vértice da pilha de vértices a visitar
        current_vertex = stack.pop()

        # Se o vértice ainda não foi visitado, adiciona à lista de visitados
        if current_vertex not in visited:
            visited.append(current_vertex)

            # Adiciona os vértices adjacentes ao vértice atual na pilha de vértices a visitar
            for i in range(len(adj_matrix)):
                if adj_matrix[current_vertex][i] != 0 and i not in visited:
                    stack.append(i)

    return visited

def roy_scc(adj_matrix):
    n = adj_matrix.shape[0]
    marcados = np.zeros(n, dtype=int)  # inicializa marcados dos vértices
    scc_list = []

    def dfs(v):
        marcados[v] = 1  # marca como visitado (cinza)
        for i in range(n):
            if adj_matrix[v, i] != 0:
                if marcados[i] == 0:  # ainda não visitado (branco)
                    dfs(i)
                elif marcados[i] == 1:  # em processo de visitação (cinza)
                    # encontrou um ciclo, não faz nada
                    pass
                else:  # já visitado (preto)
                    # adiciona todos os vértices do ciclo à lista de SCCs
                    scc = []
                    for j in range(n):
                        if marcados[j] == 2:
                            scc.append(j)
                    scc_list.append(scc)
        marcados[v] = 2  # marca como concluído (preto)

    # itera sobre todos os vértices do grafo
    for i in range(n):
        if marcados[i] == 0:  # ainda não visitado (branco)
            dfs(i)

    return scc_list

 # COLORAÇÃO DO GRAFO M2
def welsh_powell_coloring(adj_matrix):
    #pega o número de vertices do grafo
    num_vertices = len(adj_matrix)

    # Lista de vértices e suas cores
    vertex_colors = [0] * num_vertices #começa com todos os vertices com zero para depois atribuir as cores

    #priorizar vertices de mior grau
    # Ordena os vértices em ordem decrescente de grau
    vertices = sorted(range(num_vertices), key=lambda v: sum(adj_matrix[v]), reverse=True)

    # Lista de cores disponíveis
    available_colors = set()

    # Itera sobre os vértices na ordem ordenada
    for vertex in vertices:
        # Verifica as cores dos vértices adjacentes
        # Para cada vertice eu crio neighbor_colors para armazenar as cores dos vertices adjacentes
        #percorre todos os vizinhos do vertice atual e verifica se há uma conexao na matriz de adjacência
        #se tem conexão, a cor atribuída ao vizinho correspondente é adicionada ao conjunto neighbor_colors
        neighbor_colors = set(vertex_colors[neighbor] for neighbor in range(num_vertices) if adj_matrix[vertex][neighbor] != 0)

        # Encontra a primeira cor disponível
        # ou seja  procura a primeira cor disponível que ainda não foi atribuída a nenhum vizinho do vértice atual
        color = 0
        while color in neighbor_colors:
            color += 1

        # Atribui a cor ao vértice
        #A cor é armazenada na posição correspondente em vertex_colors
        vertex_colors[vertex] = color

        # E depois adiciona a cor às cores disponíveis
        available_colors.add(color)

    return vertex_colors

#VERIFICAÇÃO DE PLANARIDADE M2
def is_graph_planar(adj_matrix):
    # Cria um grafo vazio
    G = nx.Graph()

    # Adiciona as arestas do grafo com base na matriz de adjacência
    for i, row in enumerate(adj_matrix):
        for j, value in enumerate(row):
            if value == 1:
                G.add_edge(i, j)

    # Verifica se o grafo é planar se contém um subgrafo equivalente a K5  ou a K3,3
    is_planar = nx.check_planarity(G)[0]

    return is_planar


# IMPLEMENTAÇÃO A*
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def a_star(grafo, start_vertex, goal_vertex):
    adjacency_matrix = grafo.matriz
    #ler arquivo de coordenadas para ele saber a posição x y dos vertices (utilizado na distancia de manhattan)
    coordenadas = {}
    with open('coordenadas.json', 'r') as arquivo:
        # Ler o conteúdo do arquivo
        conteudo = arquivo.read()
        # Converter o JSON para dicionário Python
        coordenadas = json.loads(conteudo)

    #variavél inicializada com o número de vertices do grafo
    num_vertices = len(adjacency_matrix)

    # Lista para armazenar o caminho mínimo encontrado
    path = []

    # Conjunto aberto (que é o que eu vou visitar) /Fila de prioridade(para pegar o menor valor) para ordenar os vértices a serem explorados
    pq = PriorityQueue()

    # Dicionário para armazenar o custo total estimado de um vértice até o destino (g(n) + h(n))
    #ou seja distancia que cada vertice está do destino
    total_cost = {v: float('inf') for v in range(num_vertices)}

    # Dicionário para armazenar o custo atual do caminho do vértice de origem até o vértice atual (g(n))
    current_cost = {v: float('inf') for v in range(num_vertices)}
    current_cost[start_vertex] = 0

    # Dicionário para armazenar o vértice anterior no caminho mínimo
    previous_vertex = {v: None for v in range(num_vertices)}

    # Calcula a distância de Manhattan entre os vértices de origem e destino
    #coordenadas do vertice de partida (start) e destino (goal) são obtidas do dicionario de coordenadas
    #calculo a distancia de manhatann entre eles (v de partida e v destino)
    # h recebe resultado

    start_x, start_y = coordenadas[grafo.vertices[start_vertex]]
    goal_x, goal_y = coordenadas[grafo.vertices[goal_vertex]]
    h = manhattan_distance(start_x, start_y, goal_x, goal_y)

    # Define o custo total estimado do vértice de origem até o destino
    total_cost[start_vertex] = h

    # v partida é adicionalo a fila de prioridade com o custo total estimado e o v como uma tupla
    # Adiciona o vértice de origem na fila de prioridade
    pq.put((total_cost[start_vertex], start_vertex))

    while not pq.empty(): #roda while até que a lista de prioridade pq esteja vazia
        # Obtém o vértice com o menor custo total estimado da fila de prioridade
        _, current_vertex = pq.get()

        # Verifica se o vertice atual chegou no  vértice de destino
        if current_vertex == goal_vertex:
            # Reconstrói o caminho mínimo a partir do vértice de destino
            while current_vertex is not None:
                path.insert(0, current_vertex)
                current_vertex = previous_vertex[current_vertex]
            break

        # Explora os vértices adjacentes ao vértice atual
        for neighbor in range(num_vertices): #pega todos os vizinhos do vertice atual
            if adjacency_matrix[current_vertex][neighbor] != 0: #verifica se esse possível vizinho e matriz de adjacencia do vertice atual  é diferente de zero
               # Se diferente de zero =  há uma aresta entre o vertice atual e o possível vizinho
                # Calcula o custo atual desse vizinho( caminho do vértice de origem até o vértice adjacente)
                cost = current_cost[current_vertex] + adjacency_matrix[current_vertex][neighbor]

                # Verifica se encontrou um caminho com menor custo para o vértice adjacente
                if cost < current_cost[neighbor]:
                    # Atualiza o custo atual e o vértice anterior
                    current_cost[neighbor] = cost
                    previous_vertex[neighbor] = current_vertex

                    # Calcula a distância de Manhattan entre o vértice adjacente e o vértice de destino
                    neighbor_x, neighbor_y = coordenadas[grafo.vertices[neighbor]]
                    h = manhattan_distance(neighbor_x, neighbor_y, goal_x, goal_y)

                    # Define o custo total estimado do vértice adjacente até o destino
                    total_cost[neighbor] = cost + h

                    pq.put((total_cost[neighbor], neighbor))
    return path

class Application(tk.Frame): #Criando o layout
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Grafo")
        self.master.geometry("500x400")
        self.master.resizable(False, False)
        self.create_widgets()
        #criando um grafo padrão
        self.grafo = Grafo(['Cascavel','Toledo','Foz_do_iguacu','Francisco_beltrao','Sao_mateus_do_sul','Paranagua', 'Guarapuava', 'Londrina', 'Ponta_grossa', 'Maringa', 'Umuarama', 'Curitiba'], False, True)
        self.grafo.adicionar_aresta(0, 2, 143)
        self.grafo.adicionar_aresta(0, 6, 250)
        self.grafo.adicionar_aresta(0, 1, 50)
        self.grafo.adicionar_aresta(0, 3, 186)
        self.grafo.adicionar_aresta(3, 4, 354)
        self.grafo.adicionar_aresta(4, 11, 157)
        self.grafo.adicionar_aresta(5, 11, 90)
        self.grafo.adicionar_aresta(8, 11, 114)
        self.grafo.adicionar_aresta(6, 8, 165)
        self.grafo.adicionar_aresta(1, 10, 126)
        self.grafo.adicionar_aresta(9, 10, 190)
        self.grafo.adicionar_aresta(7, 9, 114)
        self.grafo.adicionar_aresta(8, 9, 314)
        self.grafo.adicionar_aresta(7, 8, 273)

    def create_widgets(self): #Criando os botões
        # PRIMEIRA LINHA
        self.graph_frame = tk.Frame(self.master, width=500, height=50)
        self.graph_frame.pack(side="top")

        self.chk_direcionado_value = tk.BooleanVar()
        chk1 = tk.Checkbutton(self.graph_frame, text="Direcionado", variable=self.chk_direcionado_value) #Frame representa uma linha na interface
        def checkbox_mudou():
            self.grafo.direcionado = self.chk_direcionado_value.get()
            exibe_grafo(self.grafo)
        chk1.config(command=checkbox_mudou)
        chk1.pack(side='left', padx=5, pady=5)

        self.att_button = tk.Button(self.graph_frame, text="Log Arestas", command=self.log_arestas)
        self.att_button.pack(side="left", padx=5, pady=5)

        self.att_button = tk.Button(self.graph_frame, text="Atualizar", command=self.atualizar_grafo)
        self.att_button.pack(side="left", padx=5, pady=5)

        self.reset_button = tk.Button(self.graph_frame, text="Resetar", command=self.resetar_grafo)
        self.reset_button.pack(side="left", padx=5, pady=5)

        # SEGUNDA LINHA
        self.menu_frame = tk.Frame(self.master, width=500, height=50)
        self.menu_frame.pack(side="top")

        self.add_vertex_button = tk.Button(self.menu_frame, text="Adicionar Vertice", command=self.add_vertex)
        self.add_vertex_button.pack(side="left", padx=5, pady=5)

        self.remove_vertex_button = tk.Button(self.menu_frame, text="Remover Vertice", command=self.remove_vertex)
        self.remove_vertex_button.pack(side="left", padx=5, pady=5)

        self.add_edge_button = tk.Button(self.menu_frame, text="Adicionar Aresta", command=self.add_edge)
        self.add_edge_button.pack(side="left", padx=5, pady=5)

        self.remove_edge_button = tk.Button(self.menu_frame, text="Remover Aresta", command=self.remove_edge)
        self.remove_edge_button.pack(side="left", padx=5, pady=5)

        # TERCEIRA LINHA
        self.inputs_frame = tk.Frame(self.master, width=500, height=100)
        self.inputs_frame.pack(side="top")

        self.v1_label = tk.Label(self.inputs_frame, text="Vertice 1:")
        self.v1_label.pack(side="left", padx=5, pady=5)

        self.v1_input = tk.Entry(self.inputs_frame)
        self.v1_input.pack(side="left", padx=5, pady=5)

        self.v2_label = tk.Label(self.inputs_frame, text="Vertice 2:")
        self.v2_label.pack(side="left", padx=5, pady=5)

        self.v2_input = tk.Entry(self.inputs_frame)
        self.v2_input.pack(side="left", padx=5, pady=5)

        self.peso_label = tk.Label(self.inputs_frame, text="Peso:")
        self.peso_label.pack(side="left", padx=5, pady=5)

        self.peso_input = tk.Entry(self.inputs_frame)
        self.peso_input.pack(side="left", padx=5, pady=5)

        # QUARTA LINHA
        self.alg_frame = tk.Frame(self.master, width=500, height=100)
        self.alg_frame.pack(side="top")

        self.add_bfs_button = tk.Button(self.alg_frame, text="BUSCA EM LARGURA", command=self.call_bfs)
        self.add_bfs_button.pack(side="left", padx=5, pady=5)

        self.add_dfs_button = tk.Button(self.alg_frame, text="BUSCA EM PROFUNDIDADE", command=self.call_dfs)
        self.add_dfs_button.pack(side="left", padx=5, pady=5)

        self.add_prim_button = tk.Button(self.alg_frame, text="PRIM", command=self.call_prim)
        self.add_prim_button.pack(side="left", padx=5, pady=5)

        self.add_roy_button = tk.Button(self.alg_frame, text="ROY", command=self.call_roy)
        self.add_roy_button.pack(side="left", padx=5, pady=5)

        #QUINTA LINHA
        self.alg_frame = tk.Frame(self.master, width=500, height=100)
        self.alg_frame.pack(side="top")

        self.add_roy_button = tk.Button(self.alg_frame, text="COLORAÇÃO", command= self.call_coloracao)
        self.add_roy_button.pack(side="left", padx=5, pady=5)

        self.add_roy_button = tk.Button(self.alg_frame, text="PLANARIDADE", command= self.call_planar)
        self.add_roy_button.pack(side="left", padx=5, pady=5)

        self.add_roy_button = tk.Button(self.alg_frame, text="A*", command=self.call_astar)
        self.add_roy_button.pack(side="left", padx=5, pady=5)

    def call_coloracao(self):
        # Obtém o dicionário de cores usando a função welsh_powell_coloring
        vertex_colors = welsh_powell_coloring(self.grafo.matriz)

        # Exibe o grafo colorido
        exibe_grafo(self.grafo, vertex_colors)
        print(vertex_colors)

    def call_planar(self):
        result = is_graph_planar(self.grafo.matriz)
        print(result)

    def call_astar(self):
        v1 = np.where(self.grafo.vertices == self.v1_input.get())[0][0]
        v2 = np.where(self.grafo.vertices == self.v2_input.get())[0][0]
        # Obtém o caminho mínimo usando o algoritmo A*
        path = a_star(self.grafo, v1, v2)

        if path is not None:
            # Exibe o grafo com as arestas do caminho mínimo pintadas
            exibe_grafo(self.grafo, caminho=path)
        else:
            print("Não foi possível encontrar um caminho mínimo.")

    def resetar_grafo(self):
        self.grafo = Grafo([], self.grafo.direcionado, self.grafo.ponderado)
        exibe_grafo(self.grafo)

    def add_vertex(self):
        vertice = simpledialog.askstring("Adicionar Vertice", "Digite o nome do vertice:")
        if vertice is not None:
            self.grafo.adicionar_vertice(vertice)
        exibe_grafo(self.grafo)

    def remove_vertex(self):
        vertice = simpledialog.askstring("Remover Vertice", "Digite o nome do vertice:")
        vertice = np.where(self.grafo.vertices == vertice)[0][0]
        if vertice is not None:
            self.grafo.remover_vertice(vertice)
        exibe_grafo(self.grafo)

    def atualizar_grafo(self):
        exibe_grafo(self.grafo)

    def add_edge(self):
        # pega o nome do vertices informados pelo usuario e procura o indice deles
        v1 = np.where(self.grafo.vertices == self.v1_input.get())[0][0]
        v2 = np.where(self.grafo.vertices == self.v2_input.get())[0][0]
        peso = int(self.peso_input.get())
        if self.grafo.ponderado:
            self.grafo.adicionar_aresta(v1, v2, peso)
        else:
            self.grafo.adicionar_aresta(v1, v2)
        exibe_grafo(self.grafo)

    def remove_edge(self):
        v1 = np.where(self.grafo.vertices == self.v1_input.get())[0][0]
        v2 = np.where(self.grafo.vertices == self.v2_input.get())[0][0]
        self.grafo.remover_aresta(v1, v2)
        exibe_grafo(self.grafo)

    def call_prim(self):
        # cria um grafo novo com a mesma configuracao do original
        grafo_prim = Grafo(self.grafo.vertices, False, self.grafo.ponderado)
        # resultado_prim é o conjunto de arestas e seus respectivos pesos
        resultado_prim = prim(self.grafo.matriz)
        # passa por cada uma das arestas e adiciona elas no grafo do prim para exibí-lo depois
        for item in resultado_prim:
            v1 = item[0]
            v2 = item[1]
            peso = item[2]
            grafo_prim.adicionar_aresta(v1, v2, peso)
        exibe_grafo(grafo_prim)

    def call_roy(self):
        result = roy_scc(self.grafo.matriz)
        print(result)

    def call_bfs(self):
        # cria um grafo novo com a mesma configuracao do original
        grafo_bfs = Grafo(self.grafo.vertices, True, False)
        # o resultado_bfs é uma lista com a ordem em que os vertices foram visitados
        resultado_bfs = bfs(self.grafo.matriz, 0)
        # passa por cada vertice e cria uma aresta do vertice atual (i) com o proximo vertice (i + 1)
        for i in range(len(resultado_bfs) - 1):
            v1 = resultado_bfs[i]
            v2 = resultado_bfs[i + 1]
            grafo_bfs.adicionar_aresta(v1, v2)
        exibe_grafo(grafo_bfs)

    def call_dfs(self):
        # cria um grafo novo com a mesma configuracao do original
        grafo_dfs = Grafo(self.grafo.vertices, True, False)
        # resultado_dffs é uma lista com a ordem em que os vertices foram visitados
        resultado_dfs = dfs(self.grafo.matriz, 0)
        # passa por cada vertice e cria uma aresta do vertice atual (i) com o proximo vertice (i + 1)
        for i in range(len(resultado_dfs) - 1):
            v1 = resultado_dfs[i]
            v2 = resultado_dfs[i + 1]
            grafo_dfs.adicionar_aresta(v1, v2)
        exibe_grafo(grafo_dfs)

    def log_arestas(self):
        # faz um loop que passa por cada chave do dicionario de arestas e printa o id da aresta, vertice de origem e vertice de destino
        [ print(key,':', self.grafo.vertices[self.grafo.arestas[key]['v1']], '->', self.grafo.vertices[self.grafo.arestas[key]['v2']]) for key in self.grafo.arestas ]

root = tk.Tk()
app = Application(master=root)
app.mainloop()
