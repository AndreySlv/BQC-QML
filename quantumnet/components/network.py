import networkx as nx
from qiskit import QuantumCircuit
from ..objects import Logger, Qubit
from ..components import *
from .layers import *
import random
import os
import csv
import matplotlib.pyplot as plt


class Network():
    """
    Um objeto para utilizar como rede.
    """
    def __init__(self) -> None:
        # Sobre a rede
        self._graph = nx.Graph()
        self._topology = None
        self._hosts = {}
        self.node_colors = []
        # Camadas
        self._physical = PhysicalLayer(self)
        self._link = LinkLayer(self, self._physical)
        self._network = NetworkLayer(self, self._link, self._physical)
        self._transport = TransportLayer(self, self._network, self._link, self._physical)
        self._application = ApplicationLayer(self, self._transport, self._network, self._link, self._physical)
        # Sobre a execução
        self.logger = Logger.get_instance()
        self.count_qubit = 0
        self.max_prob = 1
        self.min_prob = 0.2
        self.timeslot_total = 0
        self.qubit_timeslots = {} 
        self.requests_queue = []   
        self.final_slice_1_paths = None  
        self.final_slice_2_paths = None  

    @property
    def hosts(self):
        """
        Dicionário com os hosts da rede. No padrão {host_id: host}.

        Returns:
            dict : Dicionário com os hosts da rede.
        """
        return self._hosts
    
    @property
    def graph(self):
        """
        Grafo da rede.

        Returns:
            nx.Graph : Grafo da rede.
        """
        return self._graph
    
    @property
    def nodes(self):
        """
        Nós do grafo da rede.

        Returns:
            list : Lista de nós do grafo.
        """
        return self._graph.nodes()
    
    @property
    def edges(self):
        """
        Arestas do grafo da rede.

        Returns:
            list : Lista de arestas do grafo.
        """
        return self._graph.edges()
    
    # Camadas
    @property
    def physical(self):
        """
        Camada física da rede.

        Returns:
            PhysicalLayer : Camada física da rede.
        """
        return self._physical
    
    @property
    def linklayer(self):
        """
        Camada de enlace da rede.

        Returns:
            LinkLayer : Camada de enlace da rede.
        """
        return self._link
    
    @property 
    def networklayer(self):
        """
        Camada de rede da rede.

        Returns:
            NetworkLayer : Camada de rede da rede.
        """
        return self._network
    
    @property   
    def transportlayer(self):
        """
        Camada de transporte de transporte.

        Returns:
            TransportLayer : Camada de transporte de transporte.
        """
        return self._transport
    
    @property
    def application_layer(self):
        """
        Camada de transporte de aplicação.

        Returns:
            ApplicationLayer : Camada de aplicação.
        """
        return self._application

    def draw(self):
        """
        Desenha a rede.
        """
        nx.draw(self._graph, with_labels=True)
    
    def add_host(self, host: Host):
        """
        Adiciona um host à rede no dicionário de hosts, e o host_id ao grafo da rede.
            
        Args:
            host (Host): O host a ser adicionado.
        """
        # Adiciona o host ao dicionário de hosts, se não existir
        if host.host_id not in self._hosts:        
            self._hosts[host.host_id] = host
            Logger.get_instance().debug(f'Host {host.host_id} adicionado aos hosts da rede.')
        else:
            raise Exception(f'Host {host.host_id} já existe nos hosts da rede.')
            
        # Adiciona o nó ao grafo da rede, se não existir
        if not self._graph.has_node(host.host_id):
            self._graph.add_node(host.host_id)
            Logger.get_instance().debug(f'Nó {host.host_id} adicionado ao grafo da rede.')
            
        # Adiciona as conexões do nó ao grafo da rede, se não existirem
        for connection in host.connections:
            if not self._graph.has_edge(host.host_id, connection):
                self._graph.add_edge(host.host_id, connection)
                Logger.get_instance().debug(f'Conexões do {host.host_id} adicionados ao grafo da rede.')
    
    def get_host(self, host_id: int) -> Host:
        """
        Retorna um host da rede.

        Args:
            host_id (int): ID do host a ser retornado.

        Returns:
            Host : O host com o host_id fornecido.
        """
        return self._hosts[host_id]

    def get_eprs(self):
        """
        Cria uma lista de qubits entrelaçados (EPRs) associadas a cada aresta do grafo.

        Returns:
            Um dicionários que armazena as chaves que são as arestas do grafo e os valores são as
              listas de qubits entrelaçados (EPRs) associadas a cada aresta. 
        """
        eprs = {}
        for edge in self.edges:
            eprs[edge] = self._graph.edges[edge]['eprs']
        return eprs
    
    def get_eprs_from_edge(self, alice: int, bob: int) -> list:
        """
        Retorna os EPRs de uma aresta específica.

        Args:
            alice (int): ID do host Alice.
            bob (int): ID do host Bob.
        Returns:
            list : Lista de EPRs da aresta.
        """
        edge = (alice, bob)
        return self._graph.edges[edge]['eprs']
    
    def remove_epr(self, alice: int, bob: int) -> list:
        """
        Remove um EPR de um canal.

        Args:
            channel (tuple): Canal de comunicação.
        """
        channel = (alice, bob)
        try:
            epr = self._graph.edges[channel]['eprs'].pop(-1)   
            return epr
        except IndexError:
            raise Exception('Não há Pares EPRs.')   

    def set_topology_for_slices(self, graph_type: str, dimensions: tuple, clients: list, server: int):
        """
        Configura a topologia da rede especificamente para a simulação de slices.
        
        Args:
            graph_type (str): Tipo do grafo. Pode ser 'grade', 'linha' ou 'anel'.
            dimensions (tuple): Dimensões da topologia. Por exemplo, (4, 4) para uma grade 4x4.
            clients (list): IDs dos nós que serão configurados como clientes.
            server (int): ID do nó que será configurado como servidor.
        """
        # Cria a topologia
        if graph_type == 'grade':
            if len(dimensions) != 2:
                raise ValueError("Para topologia 'grade', são necessárias duas dimensões.")
            self._graph = nx.grid_2d_graph(*dimensions)
        elif graph_type == 'linha':
            if len(dimensions) != 1:
                raise ValueError("Para topologia 'linha', é necessário um único valor de dimensão.")
            self._graph = nx.path_graph(dimensions[0])
        elif graph_type == 'anel':
            if len(dimensions) != 1:
                raise ValueError("Para topologia 'anel', é necessário um único valor de dimensão.")
            self._graph = nx.cycle_graph(dimensions[0])
        else:
            raise ValueError(f"Tipo de grafo '{graph_type}' não suportado.")

        self._graph = nx.convert_node_labels_to_integers(self._graph)  # Converte rótulos dos nós para inteiros
        total_nodes = len(self._graph.nodes)

        # Valida os IDs de clientes e servidor
        if server >= total_nodes or any(client >= total_nodes for client in clients):
            raise ValueError("IDs de clientes ou servidor estão fora do intervalo de nós disponíveis na topologia.")

        # Configura pesos nas arestas
        for edge in self._graph.edges:
            self._graph.edges[edge]['weight'] = 1

        # Inicializa os nós como ServerNode, ClientNode ou RegularNode
        self._hosts = {}
        self.node_colors = []

        for node in self._graph.nodes:
            if node == server:
                self._hosts[node] = ServerNode(node)
                self.node_colors.append('green')  # Servidor
            elif node in clients:
                self._hosts[node] = ClientNode(node)
                self.node_colors.append('red')  # Clientes
            else:
                self._hosts[node] = RegularNode(node)
                self.node_colors.append('#1f78b8')  # Nós regulares

        # Inicializa canais e EPRs
        self.start_hosts()
        self.start_channels()
        self.start_eprs()

        # Log e confirmação
        self.logger.log(f"Topologia configurada: {graph_type} ({dimensions}) com {len(clients)} clientes e 1 servidor.")
        print("Topologia configurada com sucesso para slices!")

    def calculate_paths(self, clients, server):
        """
        Calcula os caminhos para cada cliente, com cada cliente correspondendo a um slice.

        Args:
            clients (list): IDs dos nós clientes.
            server (int): ID do nó servidor.

        Returns:
            list: Lista contendo os caminhos para cada cliente (slice).
        """
        slice_paths = []  # Cada cliente será um slice
        edge_weights = nx.get_edge_attributes(self._graph, 'weight')

        for client in clients:
            # Caminho mais curto do cliente para o servidor
            path = nx.shortest_path(self._graph, source=client, target=server, weight='weight')
            slice_paths.append(path)

            # Penaliza as arestas para evitar overlaps em slices futuros
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                edge_weights[edge] = edge_weights.get(edge, 1) + 10
                reverse_edge = (edge[1], edge[0])
                if reverse_edge in edge_weights:
                    edge_weights[reverse_edge] += 10

            # Atualiza os pesos no grafo após o cálculo de cada slice
            nx.set_edge_attributes(self._graph, edge_weights, 'weight')

        return slice_paths

    def visualize_slices(self, clients, server, slice_paths):
        """
        Visualiza o grafo com os caminhos para múltiplos slices distinguidos por cores.

        Args:
            clients (list): IDs dos nós clientes.
            server (int): ID do nó servidor.
            slice_paths (list): Lista de caminhos para cada cliente (slice).
        """
        pos = nx.spring_layout(self._graph)
        plt.figure(figsize=(10, 10))

        # Desenha a base do grafo
        nx.draw(self._graph, pos, with_labels=True, node_size=500, node_color="lightgray", edge_color="gray")

        nx.draw_networkx_nodes(self._graph, pos, nodelist=[server], node_color="red", label="Server")
        nx.draw_networkx_nodes(self._graph, pos, nodelist=clients, node_color="blue", label="Clients")

        # Define cores para os slices
        colors = plt.cm.tab10.colors  # Paleta de cores

        # Desenha os caminhos de cada slice
        for slice_index, path in enumerate(slice_paths):
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(self._graph, pos, edgelist=edges,
                                edge_color=colors[slice_index % len(colors)], width=2,
                                label=f"Slice {slice_index + 1}")

        plt.title("Paths for Slices")
        plt.legend()
        plt.show()

    def run_slice_simulation(self, clients, server):
        """
        Roda a simulação de slices para a topologia configurada.

        Args:
            clients (list): IDs dos nós clientes.
            server (int): ID do nó servidor.

        Returns:
            list: Lista de caminhos finais para cada cliente (slice).
        """
        # Configura pesos padrão
        for edge in self._graph.edges:
            self._graph.edges[edge]['weight'] = 1

        # Calcula os caminhos para os slices
        slice_paths = self.calculate_paths(clients, server)

        # Armazena as rotas como atributo da rede
        self.final_slice_paths = slice_paths

        print(f"Final Slice Paths for {len(clients)} slices:", self.final_slice_paths)

        # Visualiza os slices
        self.visualize_slices(clients, server, slice_paths)

        self.logger.log(f"Simulação de slices concluída para {len(clients)} clientes e servidor {server}.")

        # Retorna os caminhos para a camada de aplicação
        return self.final_slice_paths
    
        
    def set_ready_topology(self, topology_name: str, num_clients: int, *args: int, clients=None, server=None) -> None:
        """
        Cria um grafo com uma topologia pronta e inicializa os nós como servidor, clientes e normais.

        Args:
            topology_name (str): Nome da topologia.
            num_clients (int): Número de nós que serão clientes.
            *args (int): Argumentos para a topologia, geralmente o número de nós totais.
            clients (list, optional): Lista de nós que devem ser clientes.
            server (int, optional): Nó que será o servidor.
        """
        # Converter o nome da topologia para minúsculas para aceitar qualquer variação de letras
        topology_name = topology_name.lower()

        # Cria a topologia conforme o nome
        if topology_name == 'grade':
            if len(args) != 2:
                raise Exception('Para a topologia Grade, são necessários dois argumentos.')
            self._graph = nx.grid_2d_graph(*args)
        elif topology_name == 'linha':
            if len(args) != 1:
                raise Exception('Para a topologia Linha, é necessário um argumento.')
            self._graph = nx.path_graph(*args)
        elif topology_name == 'anel':
            if len(args) != 1:
                raise Exception('Para a topologia Anel, é necessário um argumento.')
            self._graph = nx.cycle_graph(*args)

        # Converte os labels dos nós para inteiros
        self._graph = nx.convert_node_labels_to_integers(self._graph)

        total_nodes = len(self._graph.nodes())
        self.node_colors = []  # Armazena as cores dos nós

        # Inicializa o nó servidor
        if server is not None:
            self._hosts[server] = ServerNode(server)
            self.node_colors.append('green')
        else:
            self._hosts[0] = ServerNode(0)
            self.node_colors.append('green')

        for edge in self._graph.edges:
            self._graph.edges[edge]['weight'] = 1  # Configura o peso padrão

        # Inicializa os clientes com base na lista fornecida
        if clients:
            for client in clients:
                if client < total_nodes:
                    self._hosts[client] = ClientNode(client)
                    self.node_colors.append('red')
                else:
                    raise ValueError(f'O nó {client} não existe na topologia de {total_nodes} nós.')

        # Inicializa os nós restantes como regulares
        for node in range(total_nodes):
            if node not in self._hosts:
                self._hosts[node] = RegularNode(node)
                self.node_colors.append('#1f78b8')

        # Inicia os hosts, canais e pares EPRs
        self.start_hosts()
        self.start_channels()
        self.start_eprs()

    def draw(self):
        node_colors = [self._hosts[node].color() for node in self._graph.nodes()]
        nx.draw(self._graph, with_labels=True, node_color=node_colors, node_size=800)
        plt.show()


    def start_hosts(self, num_qubits: int = 0):
        """
        Inicializa os hosts da rede com exceção do servidor (host 0).

        Args:
            num_qubits (int): Número de qubits a serem inicializados para cada host, exceto o host 0 (servidor).
        """
        for host_id in self._hosts:
            # Evita que o servidor (host 0) receba qubits
            if host_id == 10:
                self.logger.log(f"Host {host_id} é o servidor, não receberá qubits.")
                continue
            
            # Inicializa os qubits para os demais hosts
            for i in range(num_qubits):
                self.physical.create_qubit(host_id, increment_timeslot=False, increment_qubits=False)
            self.logger.log(f"Host {host_id} inicializado com {num_qubits} qubits.")
        
        print("Hosts inicializados")


    def start_channels(self):
        """
        Inicializa os canais da rede.
        
        Args:
            prob_on_demand_epr_create (float): Probabilidade de criar um EPR sob demanda.
            prob_replay_epr_create (float): Probabilidade de criar um EPR de replay.
        """
        for edge in self.edges:
            self._graph.edges[edge]['busy_timeslots'] = set()  # Adiciona atributo de timeslots ocupados
            self._graph.edges[edge]['prob_on_demand_epr_create'] = random.uniform(self.min_prob, self.max_prob)
            self._graph.edges[edge]['prob_replay_epr_create'] = random.uniform(self.min_prob, self.max_prob)
            self._graph.edges[edge]['eprs'] = list()
        print("Canais inicializados")
        
    def start_eprs(self, num_eprs: int = 2):
        """
        Inicializa os pares EPRs nas arestas da rede.

        Args:
            num_eprs (int): Número de pares EPR a serem inicializados para cada canal.
        """
        for edge in self.edges:
            for i in range(num_eprs):
                epr = self.physical.create_epr_pair(increment_timeslot=False,increment_eprs=False)
                self._graph.edges[edge]['eprs'].append(epr)
                self.logger.debug(f'Par EPR {epr} adicionado ao canal.')
        print("Pares EPRs adicionados")
        
    def timeslot(self):
        """
        Incrementa o timeslot da rede.
        """
        self.timeslot_total += 1
        self.apply_decoherence_to_all_layers()

    def get_timeslot(self):
        """
        Retorna o timeslot atual da rede.

        Returns:
            int : Timeslot atual da rede.
        """
        return self.timeslot_total

    def register_qubit_creation(self, qubit_id, timeslot):
        """
        Registra a criação de um qubit associando-o ao timeslot em que foi criado.
    
        Args:
            qubit_id (int): ID do qubit criado.
            timeslot (int): Timeslot em que o qubit foi criado.
        """
        self.qubit_timeslots[qubit_id] = {'timeslot': timeslot}
        
    def display_all_qubit_timeslots(self):
        """
        Exibe o timeslot de todos os qubits criados nas diferentes camadas da rede.
        Se nenhum qubit foi criado, exibe uma mensagem apropriada.
        """
        if not self.qubit_timeslots:
            print("Nenhum qubit foi criado.")
        else:
            for qubit_id, info in self.qubit_timeslots.items():
                print(f"Qubit {qubit_id} foi criado no timeslot {info['timeslot']}")
                
    def get_created_eprs(self):
        """
        Obtém o total de pares EPR criados em todas as camadas da rede.

        Returns:
            int: O número total de pares EPR criados.
        """
        total_created_eprs = (self._physical.get_created_eprs()+
                      self._link.get_created_eprs() +
                      self._network.get_created_eprs() +
                      self._application.get_created_eprs()
        )
        return total_created_eprs
    
    def get_total_useds_eprs(self):
        """
        Retorna o número total de EPRs (pares entrelaçados) utilizados na rede.

        Returns:
            int: Total de EPRs usados nas camadas física, de enlace e de rede.
        """
        total_eprs = (
                      self._link.get_used_eprs() +
                      self._network.get_used_eprs() +
                      self._application.get_used_eprs()
        )
        return total_eprs
        # self._physical.get_used_eprs()
        
    def get_total_useds_qubits(self):
        """
        Retorna o número total de qubits utilizados em toda a rede.

        Returns:
            int: Total de qubits usados nas camadas física, de enlace, transporte e aplicação.
        """
        total_qubits = (self._physical.get_used_qubits() +
                        self._link.get_used_qubits() +
                        self._transport.get_used_qubits() +
                        self._application.get_used_qubits()
                     
        )
        return total_qubits

    def get_metrics(self, metrics_requested=None, output_type="csv", file_name="metrics_output.csv"):
        """
        Obtém as métricas da rede conforme solicitado e as exporta, printa ou armazena.
        
        Args:
            metrics_requested: Lista de métricas a serem retornadas (opcional). 
                            Se None, todas as métricas serão consideradas.
            output_type: Especifica como as métricas devem ser retornadas.
                        "csv" para exportar em arquivo CSV (padrão),
                        "print" para exibir no console,
                        "variable" para retornar as métricas em uma variável.
            file_name: Nome do arquivo CSV (usado somente quando output_type="csv").
        
        Returns:
            Se output_type for "variable", retorna um dicionário com as métricas solicitadas.
        """
        # Dicionário com todas as métricas possíveis
        available_metrics = {
            "Timeslot Total": self.get_timeslot(),
            "EPRs Usados": self.get_total_useds_eprs(),
            "Qubits Usados": self.get_total_useds_qubits(),
            "Fidelidade na Camada de Transporte": self.transportlayer.avg_fidelity_on_transportlayer(),
            "Fidelidade na Camada de Enlace": self.linklayer.avg_fidelity_on_linklayer(),
            "Média de Rotas": self.networklayer.get_avg_size_routes()
        }
        
        # Se não foram solicitadas métricas específicas, use todas
        if metrics_requested is None:
            metrics_requested = available_metrics.keys()
        
        # Filtra as métricas solicitadas
        metrics = {metric: available_metrics[metric] for metric in metrics_requested if metric in available_metrics}

        # Tratamento conforme o tipo de saída solicitado
        if output_type == "print":
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
        elif output_type == "csv":
            current_directory = os.getcwd()
            file_path = os.path.join(current_directory, file_name)
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Métrica', 'Valor'])
                for metric, value in metrics.items():
                    writer.writerow([metric, value])
            print(f"Métricas exportadas com sucesso para {file_path}")
        elif output_type == "variable":
            return metrics
        else:
            raise ValueError("Tipo de saída inválido. Escolha entre 'print', 'csv' ou 'variable'.")

    def apply_decoherence_to_all_layers(self, decoherence_factor: float = 0.001):
        """
        Aplica decoerência a todos os qubits e EPRs nas camadas da rede que já avançaram nos timeslots.

        Este método ajusta a fidelidade de qubits e pares EPR para simular os efeitos da decoerência
        quântica em sistemas onde o tempo (timeslot) avança.

        Args:
            decoherence_factor (float): Fator de decoerência aplicado, que reduz a fidelidade. 
        """
        current_timeslot = self.get_timeslot()

        # Aplicar decoerência nos qubits de cada host
        for host_id, host in self.hosts.items():
            for qubit in host.memory:
                creation_timeslot = self.qubit_timeslots[qubit.qubit_id]['timeslot']
                if creation_timeslot < current_timeslot:
                    current_fidelity = qubit.get_current_fidelity()
                    new_fidelity = current_fidelity - (current_fidelity * decoherence_factor)
                    qubit.set_current_fidelity(new_fidelity)

        # Aplicar decoerência nos EPRs em todos os canais (arestas da rede)
        for edge in self.edges:
            if 'eprs' in self._graph.edges[edge]:
                for epr in self._graph.edges[edge]['eprs']:
                    current_fidelity = epr.get_current_fidelity()
                    new_fidelity = current_fidelity - (current_fidelity * decoherence_factor)
                    epr.set_fidelity(new_fidelity)

    def is_link_busy(self, node, timeslot):
        """
        Verifica se algum link conectado a um nó está ocupado no timeslot especificado.

        Args:
            node (int): O nó sendo verificado.
            timeslot (int): O timeslot a ser verificado.

        Returns:
            bool: True se algum link está ocupado, False caso contrário.
        """
        for neighbor in self._graph.neighbors(node):
            edge_data = self._graph.get_edge_data(node, neighbor)
            if 'busy_timeslots' in edge_data and timeslot in edge_data['busy_timeslots']:
                return True  # Retorna True se qualquer link do nó estiver ocupado
        return False
    
    def reserve_link(self, node, timeslot):
        """
        Reserva um link ocupando-o no timeslot especificado.
        
        Args:
            node (int): O nó atual sendo reservado.
            timeslot (int): O timeslot a ser reservado.
        """
        for neighbor in self._graph.neighbors(node):
            edge_data = self._graph.get_edge_data(node, neighbor)
            if 'busy_timeslots' not in edge_data:
                edge_data['busy_timeslots'] = set()
            edge_data['busy_timeslots'].add(timeslot)
    
    def restart_network(self):
        """
        Reinicia a rede, restaurando os EPRs e redefinindo qubits.
        """
        # Salva o estado original do log
        original_disabled_state = Logger.DISABLED
        
        # Desativa os logs temporariamente
        Logger.DISABLED = True

        try:
            # Reinicializar EPRs
            self.start_eprs(num_eprs=10)  

            # Reinicializar qubits nos hosts
            self.start_hosts(num_qubits=5)  
        finally:
            # Restaura o estado original do log
            Logger.DISABLED = original_disabled_state

        # Log para confirmar a reinicialização (aparecerá apenas se os logs estiverem ativados)
        self.logger.log("Rede reiniciada com sucesso.")


    # SIMULAÇÃO DA REDE

    def generate_random_circuit(self, num_qubits=10, num_gates=30):
        """
        Gera um circuito quântico aleatório, armazena suas instruções e exibe o circuito.
        
        Args:
            num_qubits (int): Número de qubits no circuito.
            num_gates (int): Número de operações (portas) no circuito.

        Returns:
            QuantumCircuit: O circuito quântico gerado.
        """
        # Cria o circuito quântico
        qc = QuantumCircuit(num_qubits)

        # Define as portas quânticas possíveis
        single_qubit_gates = ['h', 'x', 'y', 'z', 's', 't']
        two_qubit_gates = ['cx', 'cz', 'swap']

        # Aplica operações aleatórias
        for _ in range(num_gates):
            gate_type = random.choice(['single', 'two'])

            if gate_type == 'single':
                gate = random.choice(single_qubit_gates)
                qubit = random.randint(0, num_qubits - 1)
                getattr(qc, gate)(qubit)
            elif gate_type == 'two':
                gate = random.choice(two_qubit_gates)
                qubit1 = random.randint(0, num_qubits - 1)
                qubit2 = random.randint(0, num_qubits - 1)
                while qubit1 == qubit2:
                    qubit2 = random.randint(0, num_qubits - 1)

                if gate == 'cx':
                    qc.cx(qubit1, qubit2)
                elif gate == 'cz':
                    qc.cz(qubit1, qubit2)
                elif gate == 'swap':
                    qc.swap(qubit1, qubit2)

        # Exibe o circuito no console
        print(qc)

        # Desenha e exibe o circuito graficamente
        fig = qc.draw("mpl",style="clifford")
        plt.show()

        # Salva as instruções para log e debug
        saved_instructions = self.save_circuit_instructions(qc)
        self.logger.log(f"Circuito aleatório gerado com {num_qubits} qubits e {num_gates} portas. Instruções sobre o circuito.")
        for instr in saved_instructions:
            self.logger.log(f"Instrução: {instr}")

        circuit_depth = qc.depth()
        return qc, num_qubits, circuit_depth

    def save_circuit_instructions(self, circuit):
        """
        Salva as instruções de um circuito quântico em uma lista de dicionários.

        Args:
            circuit (QuantumCircuit): O circuito quântico cujas instruções serão salvas.

        Returns:
            list: Lista de instruções do circuito em formato de dicionário.
        """
        instructions = []
        for instruction in circuit.data:
            operation = instruction.operation.name
            qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
            instructions.append({
                'operation': operation,
                'qubits': qubits,
            })
        return instructions
    
    def generate_request(self, alice_id, bob_id, num_qubits, num_gates, protocols=None, slice_path=None,scenario=None):
        """
        Gera uma requisição de teletransporte de qubits.
        
        Args:
            alice_id (int): ID do cliente (Alice).
            bob_id (int): ID do servidor (Bob).
            num_qubits (int): Número de qubits a serem teletransportados.
            num_gates (int): Número de portas no circuito quântico.
            protocols (list, opcional): Lista de protocolos (pode conter 'AC_BQC' e/ou 'BFK_BQC').
            slice_path (list, opcional): Caminho do slice associado.
            scenario (int, opcional): Cenário para execução (1 ou 2).

        """
        # Se protocolos não forem especificados, escolhe aleatoriamente entre 'AC_BQC' e 'BFK_BQC'
        if protocols is None:
            protocols = random.choice(['AC_BQC', 'BFK_BQC'])
        elif isinstance(protocols, list) and len(protocols) == 0:
            protocols = random.choice(['AC_BQC', 'BFK_BQC'])  
        
        # Gere um circuito quântico aleatório
        quantum_circuit,_,circuit_depth = self.generate_random_circuit(num_qubits, num_gates)
        
        # Cria a requisição com os dados fornecidos
        request = {
            "alice_id": alice_id,
            "bob_id": bob_id,
            "num_qubits": num_qubits,
            "quantum_circuit": quantum_circuit,
            "circuit_depth": circuit_depth,
            "protocol": protocols, 
            "slice_path": slice_path,
            "scenario": scenario  
        }

        # Adiciona a requisição à fila
        self.requests_queue.append(request)
        self.logger.log(f"Requisição adicionada: Alice {alice_id} -> Bob {bob_id} com protocolo {protocols} e cenário {scenario}.")
        return request


    def generate_request_slice(self, alice_id, bob_id, num_qubits, num_gates, protocol=None, slice_path=None,scenario=None):
        """
        Gera uma requisição de teletransporte de qubits.

        Args:
            alice_id (int): ID do cliente (Alice).
            bob_id (int): ID do servidor (Bob).
            num_qubits (int): Número de qubits a serem teletransportados.
            num_gates (int): Número de portas no circuito quântico.
            protocol (str): Protocolo associado à requisição.
            slice_path (list): Caminho do slice associado.
            
        """
        # Gere um circuito quântico aleatório
        quantum_circuit,_, circuit_depth = self.generate_random_circuit(num_qubits, num_gates)

        # Cria a requisição com os dados fornecidos
        request = {
            "alice_id": alice_id,
            "bob_id": bob_id,
            "num_qubits": num_qubits,
            "quantum_circuit": quantum_circuit,
            "circuit_depth": circuit_depth,
            "protocol": protocol,
            "slice_path": slice_path,
            "scenario":scenario 
        }

        # Adiciona a requisição à fila
        self.requests_queue.append(request)
        self.logger.log(f"Requisição adicionada: Alice {alice_id} -> Bob {bob_id} com protocolo {protocol} e cenário {scenario}.")
        return request


    def send_requests_to_controller(self, controller):
        """
        Envia todas as requisições para o controlador e esvazia a fila de requisições.
        
        Args:
            controller (Controller): O controlador responsável por agendar as requisições.
        Returns:
            dict: Feedback do controlador com o status de cada requisição.
        """
        if hasattr(controller, 'schedule_requests'):
            feedback = controller.schedule_requests(self.requests_queue)  
            self.requests_queue.clear()  # Esvazia a fila após envio
            print("Todas as requisições foram enviadas para o controlador.")
            return feedback
        else:
            raise AttributeError("O controlador fornecido não possui o método 'schedule_requests'.")

    def execute_scheduled_requests(self, scheduled_requests, slice_paths=None):
        """
        Recebe e executa as requisições agendadas pelo controlador na rede.
        
        Args:
            scheduled_requests (dict): Dicionário de requisições agendadas por timeslot.
            slice_paths (dict, optional): Caminhos associados aos slices, se disponíveis.
        """
        for timeslot, requests in scheduled_requests.items():
            # Reinicia a rede antes de processar o timeslot atual
            self.logger.log(f"Reiniciando a rede antes de processar o timeslot {timeslot}.")
            self.restart_network()  # Corrigido para chamar diretamente o método da instância atual
            self.logger.log(f"Rede reiniciada. Timeslot atual: {timeslot}.")

            # Avança para o timeslot correspondente
            while self.get_timeslot() < timeslot:
                self.timeslot()  # Incrementa o timeslot da rede
                self.logger.log(f"Timeslot avançado para {self.get_timeslot()}.")

            # Executa as requisições do timeslot
            self.logger.log(f"Executando requisições do timeslot {timeslot}.")
            for request in requests:
                # Adiciona status à requisição
                status = self.execute_request(request, slice_paths)
                request['status'] = 'executado' if status else 'falhou'
                self.logger.log(f"Requisição {request} - Status: {request['status']}")
                
                
    def execute_request(self, request, slice_paths=None):
        """
        Executa uma requisição, enviando os detalhes para a camada de aplicação da rede.

        Args:
            request (dict): Requisição contendo informações como Alice, Bob, circuito, qubits e opcionalmente slice_path.
            slice_paths (dict, optional): Dicionário com os caminhos dos slices. Se None, tenta usar o slice_path da requisição ou cálculo automático.
        """
        alice_id = request['alice_id']
        bob_id = request['bob_id']
        num_qubits = request['num_qubits']
        protocol = request.get('protocol', 'generic') 
        quantum_circuit = request.get('quantum_circuit', None)
        circuit_depth = request.get('circuit_depth', 0)
        scenario = request.get('scenario', 1)

        self.logger.log(f"Executando requisição: Alice {alice_id} -> Bob {bob_id}, Protocolo: {protocol}")

        # Verifica se a requisição já possui um slice_path
        slice_path = request.get('slice_path', None)
        route = None

        # Valida e extrai a rota
        if slice_path:
            self.logger.log(f"Slice Path fornecido: {slice_path}")
            if isinstance(slice_path, dict):
                route = slice_path.get('path', None)  # Extrai a rota do dicionário
                if not route:
                    raise ValueError(f"O slice_path fornecido é inválido: {slice_path}")
            elif isinstance(slice_path, list):
                route = slice_path  # Já é uma lista
            else:
                raise ValueError(f"Formato inválido para slice_path: {slice_path}")
        else:
            # Se não há slice_path, calcula a rota automaticamente
            self.logger.log("Nenhum slice_path fornecido. Tentando calcular rota automaticamente.")
            route = self.networklayer.short_route_valid(alice_id, bob_id)

        if not route:
            raise ValueError("Nenhuma rota válida foi encontrada para a requisição.")

        if not isinstance(route, list):
            raise ValueError(f"Rota inválida: {route}. Esperado uma lista.")

        # Log da rota antes de continuar
        self.logger.log(f"Rota extraída para execução: {route}")

        success = False
        # Executa o protocolo específico
        try:
            if protocol == "AC_BQC":
                success = self.application_layer.run_app(
                    "AC_BQC", alice_id=alice_id, bob_id=bob_id,
                    num_qubits=num_qubits, circuit=quantum_circuit,
                    slice_path=route, scenario=scenario, circuit_depth=circuit_depth)
            elif protocol == "BFK_BQC":
                success = self.application_layer.run_app(
                    "BFK_BQC", alice_id=alice_id, bob_id=bob_id,
                    num_qubits=num_qubits, num_rounds=circuit_depth,
                    circuit=quantum_circuit, slice_path=route, scenario=scenario)
            else:
                # Simulação básica sem protocolo
                success = self.transport_layer.simple_teleport(alice_id, bob_id, num_qubits, route, scenario)
        except Exception as e:
            self.logger.log(f"Erro ao executar protocolo: {str(e)}")
            raise

        # Atualiza o status da requisição
        request['status'] = 'executado' if success else 'falhou'
        self.logger.log(f"Resultado da execução: {request['status']}")

        return success

