import random
import networkx as nx
from ..components import Network, Host, Logger
from qiskit import QuantumCircuit
from collections import defaultdict

class Controller():
    def __init__(self, network):
        """
        Inicializa o controlador com uma instância da rede.
        """
        self.network = network
        self.logger = Logger.get_instance()  
        self.pending_requests = []  
        self.scheduled_requests = {}  
        self.executed_requests = []  
        self.occupied_routes = {}  
        self.scheduled_requests_slice = defaultdict(list)
        self.slices = {}
        self.failed_requests = []
        
    def initialize_slices(self, network, clients, server, protocols, slice_paths_list):
        """
        Inicializa os slices de rede com base nos clientes, protocolos e rotas fornecidos.

        Args:
            network (object): A rede quântica onde os slices serão configurados.
            clients (list): Lista de IDs de clientes.
            server (int): ID do servidor.
            protocols (list): Lista de protocolos associados a cada cliente.
            slice_paths_list (list): Lista de caminhos (rotas) associados a cada slice.

        Raises:
            ValueError: Se o número de clientes, protocolos e rotas não for igual.

        """
        # Verifica a consistência entre o número de clientes, protocolos e rotas.
        if len(clients) != len(protocols) or len(protocols) != len(slice_paths_list):
            raise ValueError("O número de clientes, protocolos e rotas deve ser igual.")

        for i, (client, protocol, slice_paths) in enumerate(zip(clients, protocols, slice_paths_list), start=1):
            slice_id = f'slice_{i}'

            # Configura o slice com as informações fornecidas.
            self.slices[slice_id] = {
                "client": client,
                "server": server,
                "path": slice_paths,  
                "protocol": protocol,
            }

            # Log de configuração para depuração
            self.logger.log(f"Slice {slice_id} configurado com cliente {client}, servidor {server}, protocolo {protocol} e caminho {slice_paths}.")


    def create_routing_table(self, host_id: int) -> dict:
        """
        Cria uma tabela de roteamento com os caminhos mais curtos para cada nó.

        Args:
            host_id (int): ID do host para o qual criar a tabela de roteamento.

        Returns:
            dict: Dicionário de destinos para caminhos mais curtos.
        """
        shortest_paths = nx.shortest_path(self.network.graph, source=host_id)
        routing_table = {dest: path for dest, path in shortest_paths.items()}
        return routing_table
    

    def register_routing_tables(self):
        """
        Registra tabelas de roteamento para todos os nós.
        """
        for host_id in self.network.hosts:
            routing_table = self.create_routing_table(host_id)
            self.network.hosts[host_id].set_routing_table(routing_table)


    # Gerenciamento de Requisições

    def record_failed_request(self, request, reason=None):
        """
        Registra uma requisição que falhou.

        Args:
            request (dict): Detalhes da requisição que falhou.
            reason (str, optional): Razão pela qual a requisição falhou.
        """
        failed_entry = {
            'request': request.copy(),  # Garante que o estado atual da requisição seja armazenado
            'reason': reason or "Falha desconhecida",
            'route': request.get('slice_path', 'Não especificada'),
        }
        self.failed_requests.append(failed_entry)
        self.logger.log(f"Falha registrada: {failed_entry}")


    def receive_request(self, request):
        """
        Recebe uma requisição e tenta agendá-la.

        Args:
            request (dict): Dicionário com a requisição contendo informações como Alice, Bob, protocolo, etc.
        """
        self.pending_requests.append(request)
        self.logger.log(f"Requisição recebida: {request}")
        self.process_requests()
        
    def process_requests(self, max_attempts=1):
        """
        Processa as requisições pendentes, tentando agendá-las em timeslots disponíveis.

        Args:
            max_attempts (int): Número máximo de tentativas para agendar uma requisição.

        """
        self.prioritize_requests()
        attempts = 0
       
        while self.pending_requests and attempts < max_attempts:
            current_timeslot = self.network.get_timeslot()

            if current_timeslot == 0:
                self.network.timeslot()
                current_timeslot = self.network.get_timeslot()

            request = self.pending_requests[0]

            # Tenta agendar a requisição no timeslot atual.
            if self.try_schedule_request(request, current_timeslot):
                self.pending_requests.pop(0)
                attempts = 0
            else:
                # Caso não seja possível agendar, registra no logger e avança o timeslot.
                self.logger.log(f"Requisição {request} não pôde ser agendada. Avançando timeslot.")
                self.network.timeslot()
                attempts += 1


    def try_schedule_request(self, request, current_timeslot):
        """
        Tenta agendar uma requisição em um timeslot disponível ou compartilhar um existente.

        Args:
            request (dict): Requisição a ser agendada.
            current_timeslot (int): Timeslot atual.

        Returns:
            bool: True se a requisição foi agendada, False caso contrário.
        """
        alice_id = request['alice_id']
        bob_id = request['bob_id']
        route = self.network.networklayer.short_route_valid(alice_id, bob_id,increment_timeslot=False)

        if route:
            # Tentar reutilizar um timeslot existente
            if current_timeslot in self.scheduled_requests:
                if self.share_timeslot(route, current_timeslot):
                    self.reserve_route(route, current_timeslot)
                    self.scheduled_requests.setdefault(current_timeslot, []).append(request)
                    self.logger.log(f"Requisição agendada no mesmo timeslot {current_timeslot} para rota {route}.")
                    return True

            # Se não for possível reutilizar, busque o próximo disponível
            next_timeslot = self.find_next_available_timeslot(route)
            if self.is_route_available(route, next_timeslot):
                self.reserve_route(route, next_timeslot)
                self.scheduled_requests.setdefault(next_timeslot, []).append(request)
                self.logger.log(f"Requisição agendada: {request} no timeslot {next_timeslot}.")
                return True

        return False

    def share_timeslot(self, route, timeslot):
        """
        Verifica se a nova rota pode compartilhar o timeslot especificado,
        considerando apenas a última requisição na mesma rota.

        Args:
            route (list): A nova rota a ser analisada.
            timeslot (int): O timeslot existente.

        Returns:
            bool: True se a rota pode compartilhar o timeslot, False caso contrário.
        """
        # Obter as requisições do timeslot especificado
        if timeslot not in self.scheduled_requests:
            return True  # Nenhuma requisição, então pode compartilhar

        # Obter a última requisição da rota no mesmo timeslot
        for request in reversed(self.scheduled_requests[timeslot]):
            existing_route = self.network.networklayer.short_route_valid(
                request['alice_id'], request['bob_id']
            )
            # Verificar sobreposição de nós intermediários
            overlapping_nodes = set(route[:-1]).intersection(existing_route[:-1])
            if overlapping_nodes:
                return False  # Conflito encontrado
        return True

    def execute_scheduled_requests(self, timeslot):
        """
        Executa as requisições agendadas no timeslot especificado.

        Args:
            timeslot (int): O timeslot cujo agendamento deve ser executado.
        """
        if timeslot not in self.scheduled_requests:
            self.logger.log(f"Nenhuma requisição agendada no timeslot {timeslot}.")
            return

        self.logger.log(f"Executando requisições do timeslot {timeslot}.")
        for request in self.scheduled_requests[timeslot]:
            if self.execute_request_one(request):
                self.executed_requests.append({"request": request, "timeslot": timeslot})

        del self.scheduled_requests[timeslot]  # Limpa as requisições já executadas

    def execute_request_one(self, request):
        """
        Executa uma requisição específica, validando a rota.

        Args:
            request (dict): Requisição a ser executada.

        Returns:
            bool: True se a execução foi bem-sucedida, False caso contrário.
        """
        alice_id = request['alice_id']
        bob_id = request['bob_id']
        route = self.network.networklayer.short_route_valid(alice_id, bob_id)

        if route:
            success = self.network.execute_request(request)

            if success:
                self.logger.log(f"Requisição executada: {request}")
                self.release_route(route)
                return True
            else:
                self.logger.log(f"Falha ao executar requisição: {request}")
                self.record_failed_request(request)  # Registra a falha
                self.release_route(route)  # Libera a rota mesmo em caso de falha
                return False

        self.logger.log(f"Falha ao encontrar rota válida para requisição: {request}")
        self.record_failed_request(request)  # Registra a falha
        return False

    # Gerenciamento das Rotas
   
    def is_route_available(self, route, timeslot):
        """
        Verifica se uma rota está disponível para uso no timeslot especificado.

        Args:
            route (list): Lista de nós que compõem a rota (ex: [u, v, w]).
            timeslot (int): O timeslot em que a disponibilidade da rota será verificada.

        Returns:
            bool: True se a rota estiver disponível, False caso contrário.
        """
        for i in range(len(route) - 1):
            link = (route[i], route[i + 1])
            if self.occupied_routes.get(link) == timeslot:
                self.logger.log(f"Conflito: Link {link} ocupado no timeslot {timeslot}.")
                return False
        return True

    def reserve_route(self, route, timeslot):
        """
        Reserva uma rota para uso no timeslot especificado.

        Args:
            route (list): Rota a ser reservada.
            timeslot (int): Timeslot em que a rota será reservada.
        """
        for i in range(len(route) - 1):
            link = (route[i], route[i + 1])
            self.occupied_routes[link] = timeslot
        self.logger.log(f"Rota reservada: {route} no timeslot {timeslot}.")

    def release_route(self, route):
        """
        Libera a rota, permitindo seu reuso em outros timeslots.

        Args:
            route (list): Rota a ser liberada.
        """
        for i in range(len(route) - 1):
            link = (route[i], route[i + 1])
            self.occupied_routes.pop(link, None)
        self.logger.log(f"Rota liberada: {route}.")

    # Funções Auxiliares 

    def find_next_available_timeslot(self, route):
        """
        Encontra o próximo timeslot em que a rota estará completamente livre.

        Args:
            route (list): Rota a ser verificada.

        Returns:
            int: Próximo timeslot livre para a rota.
        """
        current_timeslot = self.network.get_timeslot()
        while not self.is_route_available(route, current_timeslot):
            current_timeslot += 1
        return current_timeslot
    
    def prioritize_requests(self):
        """
        Ordena as requisições pendentes com base em critérios de prioridade.

        """
        # Ordena por número de qubits e, em seguida, pelo número de instruções no circuito
        self.pending_requests.sort(key=lambda req: (req['num_qubits'], -len(req['quantum_circuit'].data)))


    def generate_schedule_report(self):
        """
        Gera um relatório das requisições processadas, agendadas e falhas.
        
        Returns:
            dict: Um dicionário contendo métricas de sucesso, falha e agendamento.
        """
        report = {
            "success": len(self.executed_requests) if self.executed_requests else 0,
            "failed": len(self.failed_requests) if self.failed_requests else 0,
            "scheduled": len(self.scheduled_requests) if self.scheduled_requests else 0,
            "failed_details": []  # Para armazenar detalhes das falhas
        }

        print("=== Relatório de Requisições ===")
        
        # Requisições executadas com sucesso
        if self.executed_requests:
            print("\nRequisições Executadas:")
            for entry in self.executed_requests:
                req = entry["request"]
                ts = entry["timeslot"]
                circuit_depth = req.get("circuit_depth", "N/A")  # Obter a profundidade do circuito
                print(f"- Alice ID: {req['alice_id']}, Bob ID: {req['bob_id']}, "
                    f"Nº de Qubits: {req['num_qubits']}, Circuit Depth: {circuit_depth}, "
                    f"Timeslot: {ts}")

        # Requisições agendadas
        if self.scheduled_requests:
            print("\nRequisições Agendadas:")
            for ts, requests in self.scheduled_requests.items():
                print(f"Timeslot {ts}:")
                for req in requests:
                    circuit_depth = req.get("circuit_depth", "N/A")  # Obter a profundidade do circuito
                    print(f"- Alice ID: {req['alice_id']}, Bob ID: {req['bob_id']}, "
                        f"Nº de Qubits: {req['num_qubits']}, Circuit Depth: {circuit_depth}")

        # Requisições que falharam
        if self.failed_requests:
            print("\nRequisições que falharam:")
            for failure in self.failed_requests:
                req = failure['request']  # Detalhes da requisição
                reason = failure.get('reason', 'Motivo não especificado')
                route = failure.get('route', 'Não especificada')
                circuit_depth = req.get("circuit_depth", "N/A")  # Obter a profundidade do circuito
                print(f"- Alice ID: {req['alice_id']}, Bob ID: {req['bob_id']}, "
                    f"Nº de Qubits: {req['num_qubits']}, Circuit Depth: {circuit_depth}, "
                    f"Rota: {route}, Motivo: {reason}")
                report["failed_details"].append({
                    "alice_id": req['alice_id'],
                    "bob_id": req['bob_id'],
                    "num_qubits": req['num_qubits'],
                    "circuit_depth": circuit_depth,  # Incluído no relatório detalhado
                    "route": route,
                    "reason": reason
                })

        print("\n=== Fim do Relatório ===")
        
        return report

    def send_scheduled_requests(self):
        """
        Executa todas as requisições agendadas em sequência, reiniciando a rede após cada timeslot para evitar decoerência.

        """
        self.logger.log("Iniciando execução das requisições agendadas.")
        for ts in sorted(self.scheduled_requests.keys()):
            self.logger.log(f"Processando timeslot {ts}.")
            
            # Executa as requisições do timeslot
            self.execute_scheduled_requests(ts)

            # Após a execução, reinicia a rede
            self.logger.log(f"Estado da rede antes da reinicialização: Timeslot {self.network.get_timeslot()}.")
            self.network.restart_network()
            self.logger.log(f"Rede reiniciada. Timeslot reiniciado para {self.network.get_timeslot()}.")


    # SIMULAÇÃO EM SLICES

    def schedule_requests(self, requests, slice_paths=None, protocols=None):
        """
        Agenda as requisições em timeslots com base nos slices e protocolos fornecidos. Garante que todas as requisições tenham um protocolo e que correspondam a um slice definido.

        Args:
            requests (list): Lista de requisições a serem agendadas. Cada requisição deve conter informações como `alice_id` e `protocol`.
            slice_paths (dict): Dicionário contendo os slices com informações de clientes,caminhos e protocolos.
            protocols (list): Lista de protocolos válidos para verificação.

        Returns:
            dict: Um dicionário onde a chave é o timeslot e o valor é a lista de requisições 
                atribuídas a esse timeslot.

        Raises:
            ValueError: Se `protocols` ou `slice_paths` não forem fornecidos.
                        Se uma requisição não tiver protocolo ou não corresponder a um slice.
        """
        if protocols is None or slice_paths is None:
            raise ValueError("Protocolos e slice_paths devem ser fornecidos.")

        scheduled_timeslots = {}
        current_timeslot = 1

        # Mapeia as requisições para os slices corretos
        for request in requests:
            protocol = request.get('protocol')
            if not protocol:
                raise ValueError(f"Requisição sem protocolo: {request}")
            
            # Encontra o slice correspondente para a requisição.
            slice_key = None
            for key, slice_data in slice_paths.items():
                if slice_data["client"] == request["alice_id"] and slice_data["protocol"] == protocol:
                    slice_key = key
                    break

            if slice_key:
                request['slice_path'] = slice_paths[slice_key]["path"]  # Vincula o caminho correto
            else:
                raise ValueError(f"Nenhum slice encontrado para a requisição: {request}")

        # Agrupa as requisições por timeslot
        num_slices = len(slice_paths)
        for i in range(0, len(requests), num_slices):
            scheduled_timeslots[current_timeslot] = requests[i:i + num_slices]
            current_timeslot += 1

        return scheduled_timeslots


    def map_requests_to_slices(self, requests, protocol_to_slice):
        """
        Mapeia as requisições para slices com base no protocolo.

        Args:
            requests (list): Lista de requisições.
            protocol_to_slice (dict): Dicionário que associa protocolos a slices.

        Returns:
            dict: Requisições separadas por slices.
        """
        slice_requests = {}

        for request in requests:
            protocol = request.get('protocol')
            slice_id = protocol_to_slice.get(protocol)
            if not slice_id:
                raise ValueError(f"Protocolo {protocol} não mapeado para nenhum slice.")

            if slice_id not in slice_requests:
                slice_requests[slice_id] = []
            slice_requests[slice_id].append(request)

        return slice_requests
    

    def schedule_requests_in_timeslots(self, slice_requests):
        """
        Agenda as requisições em timeslots alternando entre os slices.

        Args:
            slice_requests (dict): Requisições separadas por slices.

        Returns:
            dict: Dicionário de timeslots com requisições agendadas.
        """
        scheduled_timeslots = {}
        current_timeslot = 1

        while any(slice_requests.values()):
            current_slot_requests = []

            # Alterna entre slices para agendar as requisições
            for slice_id, requests in slice_requests.items():
                if requests:
                    current_slot_requests.append(requests.pop(0))

            if current_slot_requests:
                scheduled_timeslots[current_timeslot] = current_slot_requests
                current_timeslot += 1

        return scheduled_timeslots
    
    def print_report(self, scheduled_timeslots, slice_paths=None):
        """
        Gera um relatório detalhado das requisições processadas.

        Args:
            scheduled_timeslots (dict): Requisições agendadas por timeslot.
            slice_paths (dict, optional): Caminhos associados aos slices.

        Returns:
            dict: Contagem de sucessos e falhas.
        """
        print("\n=== Relatório de Requisições Executadas ===")
        total_success = 0
        total_failed = 0

        for timeslot, requests in scheduled_timeslots.items():
            print(f"\nTimeslot {timeslot}:")
            for request in requests:
                status = request.get('status', 'pendente')
                slice_path = request.get('slice_path', 'Não especificado')
                circuit_depth = request.get('circuit_depth', 'N/A')

                # Contar sucessos e falhas
                if status == 'executado':
                    total_success += 1
                elif status == 'falhou':
                    total_failed += 1

                print(f"- Requisição: Alice {request.get('alice_id', 'Desconhecido')} -> Bob {request.get('bob_id', 'Desconhecido')}, "
                    f"Protocolo: {request.get('protocol', 'Desconhecido')}, Nº de Qubits: {request.get('num_qubits', 'Desconhecido')}, "
                    f"Circuit Depth: {circuit_depth}, Slice Path: {slice_path}, Status: {status}")

        print("\nResumo:")
        print(f"Total de sucessos: {total_success}")
        print(f"Total de falhas: {total_failed}")
        print("\n=== Fim do Relatório ===")

        return {
            "success_count": total_success,
            "failure_count": total_failed
        }







