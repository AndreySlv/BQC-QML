import random
import math
from quantumnet.components import Host
from quantumnet.objects import Qubit, Logger
from quantumnet.components.simulation_qml import QuantumMLSimulator 
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import Session,Sampler
from qiskit import QuantumCircuit, transpile

class ApplicationLayer:
    def __init__(self, network, transport_layer, network_layer, link_layer, physical_layer):
        """
        Inicializa a camada de aplicação.
        
        args:
            network : Network : Rede.
            transport_layer : TransportLayer : Camada de Transporte 
            network_layer : NetworkLayer : Camada de rede.
            link_layer : LinkLayer : Camada de enlace.
            physical_layer : PhysicalLayer : Camada física.
        """
        self._network = network
        self._physical_layer = physical_layer
        self._network_layer = network_layer
        self._link_layer = link_layer
        self._transport_layer = transport_layer
        self.logger = Logger.get_instance()
        self.used_qubits = 0
        self.used_eprs = 0
        self.route_fidelities = []  # Armazena as fidelidades médias de cada rota

    def __str__(self):
        """ Retorna a representação em string da camada de aplicação. 
        
        returns:
            str : Representação em string da camada de aplicação."""
        return 'Application Layer'
    
    def get_used_qubits(self):
        """
        Retorna a lista de pares EPRs usados na camada de aplicação.

        Returns:
            list: Lista de pares EPRs usados.
        """
        self.logger.debug(f"Qubits usados na camada {self.__class__.__name__}: {self.used_qubits}")
        return self.used_qubits
    
    def get_used_eprs(self):
        """
        Retorna a lista de pares EPRs usados na camada de aplicação.

        Returns:
            list: Lista de pares EPRs usados.
        """
        self.logger.debug(f"Eprs usados na camada {self.__class__.__name__}: {self.used_eprs}")
        return self.used_eprs
    
    def run_app(self, app_name, alice_id, bob_id, **kwargs):
        num_qubits = kwargs.get('num_qubits', 10)
        num_rounds = kwargs.get('num_rounds', None)
        slice_path = kwargs.get('slice_path', None)  
        scenario = kwargs.get('scenario',None)
        circuit_depth = kwargs.get('circuit_depth', None) 
        modelo_qml = kwargs.get("modelo_qml",None)

        if app_name == "QKD_E91":
            return self.qkd_e91_protocol(alice_id, bob_id, num_qubits)
        elif app_name == "AC_BQC":
            return self.run_andrews_childs_protocol(alice_id, bob_id, num_qubits, slice_path=slice_path, scenario=scenario,circuit_depth=circuit_depth,modelo_qml=modelo_qml)
        elif app_name == "BFK_BQC":
            return self.bfk_protocol(alice_id, bob_id, num_qubits, num_rounds, slice_path=slice_path, scenario=scenario,circuit_depth=circuit_depth)
        else:
            self.logger.log("Aplicação não realizada ou não encontrada.")
            return False

    # PROTOCOLO E91 - QKD 

    def qkd_e91_protocol(self, alice_id, bob_id, num_bits):
        """
        Implementa o protocolo E91 para a Distribuição Quântica de Chaves (QKD).

        Args:
            alice_id (int): ID do host de Alice.
            bob_id (int): ID do host de Bob.
            num_bits (int): Número de bits para a chave.

        Returns:
            list: Chave final gerada pelo protocolo, ou None se houver falha na transmissão.
        """
        alice = self._network.get_host(alice_id)  # Obtém o host de Alice
        bob = self._network.get_host(bob_id)  # Obtém o host de Bob

        final_key = []  # Inicializa a chave final
   
        while len(final_key) < num_bits:
            num_qubits = int((num_bits - len(final_key)) * 2)  # Calcula o número de qubits necessários
            self.used_qubits += num_qubits
            self.logger.log(f'Iniciando protocolo E91 com {num_qubits} qubits.')

            # Alice prepara os qubits
            key = [random.choice([0, 1]) for _ in range(num_qubits)]  # Gera uma chave aleatória de bits
            bases_alice = [random.choice([0, 1]) for _ in range(num_qubits)]  # Gera bases de medição aleatórias para Alice
            qubits = self.prepare_e91_qubits(key, bases_alice)  # Prepara os qubits com base na chave e nas bases
            self.logger.log(f'Qubits preparados com a chave: {key} e bases: {bases_alice}')

            # Transmissão dos qubits de Alice para Bob
            success = self._transport_layer.run_transport_layer(alice_id, bob_id, num_qubits)
            if not success:
                self.logger.log(f'Falha na transmissão dos qubits de Alice para Bob.')
                return None

            self.logger.debug(f"Timeslot incrementado após transmissão: {self._network.get_timeslot()}")

            # Bob escolhe bases aleatórias e mede os qubits
            bases_bob = [random.choice([0, 1]) for _ in range(num_qubits)]  # Gera bases de medição aleatórias para Bob
            results_bob = self.apply_bases_and_measure_e91(qubits, bases_bob)  # Bob mede os qubits usando suas bases
            self.logger.log(f'Resultados das medições: {results_bob} com bases: {bases_bob}')

            # Alice e Bob compartilham suas bases e encontram os índices comuns
            common_indices = [i for i in range(len(bases_alice)) if bases_alice[i] == bases_bob[i]]  # Índices onde as bases coincidem
            self.logger.log(f'Índices comuns: {common_indices}')

            # Extração da chave com base nos índices comuns
            shared_key_alice = [key[i] for i in common_indices]  # Chave compartilhada gerada por Alice
            shared_key_bob = [results_bob[i] for i in common_indices]  # Chave compartilhada gerada por Bob

            # Verificação se as chaves coincidem
            for a, b in zip(shared_key_alice, shared_key_bob):
                if a == b and len(final_key) < num_bits:  # Limita o tamanho da chave final
                    final_key.append(a)

            self.logger.log(f"Chaves obtidas até agora: {final_key}")

            if len(final_key) >= num_bits:
                final_key = final_key[:num_bits] 
                self.logger.log(f"Protocolo E91 bem-sucedido. Chave final compartilhada: {final_key}")
                return final_key

        return None

    def prepare_e91_qubits(self, key, bases):
        """
        Prepara os qubits de acordo com a chave e as bases fornecidas para o protocolo E91.

        Args:
            key (list): Chave contendo a sequência de bits.
            bases (list): Bases usadas para medir os qubits.

        Returns:
            list: Lista de qubits preparados.
        """
        self.logger.debug(f"Timeslot incrementado na função prepare_e91_qubits: {self._network.get_timeslot()}")
        qubits = []
        for bit, base in zip(key, bases):
            qubit = Qubit(qubit_id=random.randint(0, 1000))  # Cria um novo qubit com ID aleatório
            if bit == 1:
                qubit.apply_x()  # Aplica a porta X (NOT) ao qubit se o bit for 1
            if base == 1:
                qubit.apply_hadamard()  # Aplica a porta Hadamard ao qubit se a base for 1
            qubits.append(qubit)  # Adiciona o qubit preparado à lista de qubits
        return qubits
    

    def apply_bases_and_measure_e91(self, qubits, bases):
        """
        Aplica as bases de medição e mede os qubits no protocolo E91.

        Args:
            qubits (list): Lista de qubits a serem medidos.
            bases (list): Lista de bases a serem aplicadas para a medição.

        Returns:
            list: Resultados das medições.
        """
        self.logger.debug(f"Timeslot incrementado na função apply_bases_and_measure_e91: {self._network.get_timeslot()}")
        results = []
        for qubit, base in zip(qubits, bases):
            if base == 1:
                qubit.apply_hadamard()  # Aplica a porta Hadamard antes de medir, se a base for 1
            measurement = qubit.measure()  
            results.append(measurement)  # Adiciona o resultado da medição à lista de resultados
        return results
    
    #PROTOCOLO ANDREWS CHILDS - BQC

    def run_andrews_childs_protocol(self, alice_id, bob_id, num_qubits, circuit_depth=None, slice_path=None, scenario=1, modelo_qml=None):
        """
        Executa o protocolo Andrew Childs, onde Alice prepara qubits, envia para Bob, e Bob realiza operações.

        args:
            alice_id : int : ID de Alice.
            bob_id : int : ID de Bob.
            num_qubits : int : Número de qubits a serem transmitidos.
            slice_path : list : Caminho da rota (opcional).
            scenario : int : Define o cenário do transporte (1 ou 2).
        """
        alice = self._network.get_host(alice_id)
        bob = self._network.get_host(bob_id)
        
        if circuit_depth is None:
            raise ValueError("Erro: 'circuit_depth' não foi fornecido ou está inválido.")

        self.logger.log(f"Timeslot {self._network.get_timeslot()}: Iniciando protocolo Andrew Childs entre Alice {alice_id} e Bob {bob_id}.")
        
        # Limpar memórias de Alice e Bob antes de começar
        self.logger.log("Limpando a memória do cliente (Alice) antes de iniciar o protocolo.")
        alice.memory.clear()
        self.logger.log("Limpando a memória do servidor (Bob) antes de iniciar o protocolo.")
        bob.memory.clear()

        # O cliente prepara qubits e armazena-os
        qubits = [Qubit(qubit_id=random.randint(0, 1000)) for _ in range(num_qubits)]
        self.logger.log(f"Cliente criou {len(qubits)} qubits para a transmissão.")

        # Registrar qubits no dicionário de timeslots
        for qubit in qubits:
            self._network.qubit_timeslots[qubit.qubit_id] = {'timeslot': self._network.get_timeslot()}
            self.logger.log(f"Qubit {qubit.qubit_id} registrado no timeslot {self._network.get_timeslot()}")

        # Log dos qubits após criação
        for qubit in qubits:
            self.logger.log(f"Qubit {qubit.qubit_id} criado pelo Cliente - Estado: {qubit._qubit_state}, Fase: {qubit._phase}")

        # Armazena os qubits criados na memória do cliente
        alice.memory.extend(qubits)
        self.logger.log(f"Alice recebeu {len(qubits)} qubits. Total: {len(alice.memory)} qubits na memória.")

        # Cria mensagem clássica com instruções
        operations_classical_message = [self.generate_random_operation() for _ in qubits]
        self.logger.log(f"Instruções clássicas enviadas pelo Cliente: {operations_classical_message}")

        # Calcula a rota se não fornecida
        route = slice_path or self._network.networklayer.short_route_valid(alice_id, bob_id)
        if not route:
            self.logger.log(f"Erro: Nenhuma rota encontrada entre {alice_id} e {bob_id}.")
            return None

        self.logger.log(f"Rota calculada para o transporte: {route}")

        # Limpar pares EPRs residuais na rota antes de iniciar o protocolo
        self.logger.log(f"Timeslot {self._network.get_timeslot()}: Limpando pares EPRs residuais antes de iniciar o protocolo.")
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            self._physical_layer.remove_all_eprs_from_channel((u, v))
            self.logger.log(f"Pares EPRs limpos no segmento {u} -> {v}.")

        # Transporte de Alice para Bob
        success = self._transport_layer.run_transport_layer_eprs(alice_id, bob_id, len(qubits), route=route, scenario=scenario)
        if not success:
            self.logger.log("Falha ao enviar os qubits para o servidor.")
            return None

        alice.memory.clear()
        self.logger.log(f"Cliente enviou {len(qubits)} qubits para o Servidor.")
        self.logger.log(f"Servidor tem {len(bob.memory)} qubits na memória após a recepção.")

        # === Inicia o VQC enquanto Bob faz operações ===
        self.logger.log(f"Iniciando treinamento do {modelo_qml.upper()}.")
        qml_simulator = QuantumMLSimulator()

        if modelo_qml == 'vqc':
            qml_simulator.iniciar_treinamento_vqc(max_iter=100)
            resultados = None
        elif modelo_qml == 'qcnn':
            qml_simulator.iniciar_qcnn(num_images=50, max_iter=100)
            resultados = None  

        # Servidor aplica operações
        tempo_de_operacao = circuit_depth
        print(f"Tempo de Operação: {tempo_de_operacao}")

        for _ in range(tempo_de_operacao):
            self._network.timeslot()
            self.logger.log(f"Timeslot {self._network.get_timeslot()}: Servidor aplicando operações nos qubits.")
        
        for qubit, operation in zip(qubits, operations_classical_message):
            self.apply_operation_from_message(qubit, operation)
        self.logger.log("Servidor aplicou as operações instruídas pelo Cliente nos qubits.")

        # Log após operações
        for qubit in qubits:
            self.logger.log(f"Qubit {qubit.qubit_id} após operações de Servidor - Estado: {qubit._qubit_state}, Fase: {qubit._phase}")
        
        # Limpa a memória do Cliente antes de devolver os qubits
        self.logger.log(f"Limpando a memória do cliente antes de receber os qubits devolvidos.")
        alice.memory.clear()

        # Devolve os qubits para Alice
        route_back = route[::-1]
        success = self._transport_layer.run_transport_layer_eprs(bob_id, alice_id, len(qubits), route=route_back, is_return=True, scenario=scenario)
        if not success:
            self.logger.log(f"Falha ao devolver os qubits para o cliente. O servidor tinha {len(qubits)} qubits.")
            return None

        # Evita duplicação ao adicionar os qubits devolvidos
        existing_qubits_ids = {qubit.qubit_id for qubit in alice.memory}
        new_qubits = [qubit for qubit in qubits if qubit.qubit_id not in existing_qubits_ids]
        alice.memory.extend(new_qubits)
        self.logger.log(f"Servidor devolveu {len(new_qubits)} qubits para o cliente.")

        # Log após retorno
        for qubit in qubits:
            self.logger.log(f"Qubit {qubit.qubit_id} devolvido para o cliente - Estado: {qubit._qubit_state}, Fase: {qubit._phase}")

        # Decodificação Clifford
        for qubit, operation in zip(qubits, operations_classical_message):
            self.apply_clifford_decoding(qubit, operation)
            self.logger.log(f"Cliente aplicou a decodificação Clifford no qubit {qubit.qubit_id}.")

        # Verificação final
        if len(alice.memory) == num_qubits:
            self.logger.log(f"Protocolo concluído com sucesso. O cliente tem {len(alice.memory)} qubits decodificados.")
        else:
            self.logger.log(f"Erro: Cliente tem {len(alice.memory)} qubits, mas deveria ter {num_qubits} qubits.")
            return None

        if modelo_qml == "VQC":
            resultados = qml_simulator.pegar_resultados_vqc()
            if resultados:
                self.logger.log(f"Resultados do VQC: Treino={round(resultados['accuracy_train'] * 100, 2)}%, Teste={round(resultados['accuracy_test'] * 100, 2)}%, Tempo={resultados['duration']}s")
            else:
                self.logger.log("VQC não retornou resultados válidos.")

        elif modelo_qml == "QCNN":
            resultados = qml_simulator.pegar_resultados_qcnn()
            if resultados:
                self.logger.log(f"Resultados do QCNN: Treino={round(resultados['accuracy_train'] * 100, 2)}%, Teste={round(resultados['accuracy_test'] * 100, 2)}%")
            else:
                self.logger.log("QCNN não retornou resultados válidos.")

        return qubits
    
    def generate_random_operation(self):
        """
        Gera uma operação quântica aleatória (X, Y, Z).

        Returns:
            str : Operação escolhida aleatoriamente.
        """
        operations = ['X', 'Y', 'Z']
        return random.choice(operations)

    def apply_operation_from_message(self, qubit, operation):
        """
        Aplica a operação quântica especificada em um qubit.

        Args:
            qubit : Qubit : O qubit ao qual a operação será aplicada.
            operation : str : Operação (X, Y ou Z) a ser aplicada.
        """
        if operation == 'X':
            qubit.apply_x()
        elif operation == 'Y':
            qubit.apply_y()
        elif operation == 'Z':
            qubit.apply_z()

    def apply_clifford_decoding(self, qubit, operation):
        """
        Aplica a operação Clifford de decodificação em um qubit.

        Args:
            qubit : Qubit : O qubit ao qual a operação será aplicada.
            operation : str : Operação Clifford a ser aplicada (X, Y ou Z).
        """
        if operation == 'X':
            qubit.apply_x()
        elif operation == 'Y':
            qubit.apply_y()
        elif operation == 'Z':
            qubit.apply_z()

    # PROTOCOLO BFK - BQC

    def bfk_protocol(self, client_id, server_id, num_qubits, num_rounds, circuit_depth=None, slice_path=None, scenario=1):
        """
        Executa o protocolo BFK completo: cliente prepara qubits, servidor cria brickwork e cliente envia instruções.
        
        Args:
            client_id (int): ID do cliente.
            server_id (int): ID do servidor.
            num_qubits (int): Número de qubits preparados pelo cliente.
            num_rounds (int): Número de rodadas de computação.
            slice_path (list, optional): Caminho específico para o transporte.
            scenario (int, optional): Define o cenário de simulação (1 ou 2). Default: 1.
            
        Returns:
            list: Resultados finais das medições realizadas pelo servidor.
        """
        if num_rounds is None:
            num_rounds = circuit_depth if circuit_depth is not None else num_qubits

        self.logger.log(f"Protocolo configurado para {num_rounds} rodadas.")

        print(f"Tempo de Operação: {circuit_depth}")
        
        self._network.timeslot()
        self.logger.log(f"Timeslot {self._network.get_timeslot()}. Iniciando protocolo BFK com {num_qubits} qubits, {num_rounds} rodadas, e cenário {scenario}.")

        self.used_qubits += num_qubits

        # Limpar a memória de Alice (cliente)
        client = self._network.get_host(client_id)
        if hasattr(client, 'memory') and isinstance(client.memory, list):
            client.memory.clear()
            self.logger.log(f"Memória do cliente {client_id} (Alice) limpa com sucesso.")
        else:
            self.logger.log(f"O cliente {client_id} não possui memória ou atributo 'memory' para limpar.")

        # Limpar a memória de Bob (servidor)
        server = self._network.get_host(server_id)
        if hasattr(server, 'memory') and isinstance(server.memory, list):
            server.memory.clear()
            self.logger.log(f"Memória do servidor {server_id} (Bob) limpa com sucesso.")
        else:
            self.logger.log(f"O servidor {server_id} não possui memória ou atributo 'memory' para limpar.")

        # Cliente prepara os qubits
        self._network.timeslot()
        self.logger.log(f"Timeslot {self._network.get_timeslot()}.")
        qubits = self.prepare_qubits(client_id, num_qubits)
        
        # Determinar a rota
        if slice_path:
            self.logger.log(f"Usando rota específica para o transporte: {slice_path}")
            route = slice_path
        else:
            self.logger.log(f"Calculando rota padrão para o transporte.")
            route = self._network.networklayer.short_route_valid(client_id, server_id)
            if not route:
                self.logger.log(f"Erro: Nenhuma rota encontrada entre {client_id} e {server_id}.")
                return None

        # Limpar pares EPRs residuais na rota
        self.logger.log(f"Limpando pares EPRs residuais na rota: {route}")
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            self._physical_layer.remove_all_eprs_from_channel((u, v))
            self.logger.log(f"Pares EPRs limpos no segmento {u} -> {v}.")

        # Executar a transmissão usando a rota definida
        success = self._transport_layer.run_transport_layer_eprs_bfk(client_id, server_id, num_qubits, route=route, scenario=scenario)
        if not success:
            self.logger.log(f"Falha ao transmitir qubits do cliente {client_id} para o servidor {server_id}.")
            return None

        # Servidor cria o estado de brickwork com os qubits recebidos
        self._network.timeslot()
        self.logger.log(f"Timeslot {self._network.get_timeslot()}.")
        success = self.create_brickwork_state(server_id, qubits)
        if not success:
            self.logger.log(f"Falha na criação do estado de brickwork no servidor {server_id}.")
            return None

        # Cliente instrui o servidor a medir os qubits em cada rodada
        results = self.run_computation(client_id, server_id, num_rounds, qubits)

        self.logger.log(f"Protocolo BFK concluído com sucesso. Resultados: {results}")
        return results


    def prepare_qubits(self, alice_id, num_qubits):
        """
        Prepara uma lista de qubits para o cliente especificado.

        Args:
            alice_id (int): ID do cliente que está preparando os qubits.
            num_qubits (int): Número de qubits que devem ser preparados.

        Returns:
            list: Lista de objetos Qubit preparados.

        Raises:
            AssertionError: Se o número de qubits preparados não corresponder ao valor esperado.
        """
        qubits = []

        # Loop para criar e preparar os qubits.
        for _ in range(num_qubits):
            r_j = random.choice([0, 1])  # Cliente gera um bit aleatório r_j
            qubit = Qubit(qubit_id=random.randint(0, 1000))  # Cria um qubit com ID aleatório
            if r_j == 1:
                qubit.apply_x()  # Aplica a porta X se r_j for 1
            qubits.append(qubit)
            self.logger.log(f"Qubit {qubit.qubit_id} preparado pelo cliente {alice_id}.")
        assert len(qubits) == num_qubits, "Número de qubits preparados não corresponde ao esperado."
        return qubits
    

    def create_brickwork_state(self, bob_id, qubits):
        """
        O servidor cria o estado de brickwork utilizando os qubits recebidos.

        Args:
            bob_id (int): ID do servidor que cria o estado.
            qubits (list): Lista de qubits recebidos do cliente.

        Returns:
            bool: True se o estado de brickwork foi criado com sucesso, False caso contrário.
        """
        server = self._network.get_host(bob_id)

        # Aplica a fase controlada nos qubits para criar o estado de brickwork
        for i in range(len(qubits) - 1):
            control_qubit = qubits[i]  # Qubit de controle
            target_qubit = qubits[i + 1]  # Qubit alvo
            target_qubit.apply_controlled_phase(control_qubit)  # Aplica a fase controlada
        self.logger.log(f"Servidor {bob_id} criou um estado de brickwork com {len(qubits)} qubits.")
        return True

    
    def run_computation(self, alice_id, bob_id, num_rounds, qubits):
        """
        Cliente instrui o servidor a realizar medições em todos os qubits durante as rodadas de computação.

        Args:
            alice_id (int): ID do cliente que fornece instruções.
            bob_id (int): ID do servidor que realiza as medições.
            num_rounds (int): Número de rodadas de computação a serem executadas.
            qubits (list): Lista de qubits a serem medidos.

        Returns:
            list: Resultados das medições realizadas pelo servidor em todas as rodadas.
        """
        client = self._network.get_host(alice_id)
        server = self._network.get_host(bob_id)
        measurement_results = []

        # Inicializa os ângulos de medição para todos os qubits
        angles = [random.uniform(0, 2 * math.pi) for _ in qubits]
        self.logger.log(f"Cliente {alice_id} inicializou ângulos de medição: {angles}")

        # Executa as rodadas de computação
        for round_num in range(num_rounds):
            round_results = []

            # Medição de todos os qubits na rodada atual
            for i, qubit in enumerate(qubits):
                theta = angles[i]
                self.logger.log(f"Rodada {round_num + 1}: Cliente {alice_id} instrui o servidor a medir o qubit {qubit.qubit_id} na base {theta}.")
                
                # Servidor realiza a medição
                self._network.timeslot()
                result = qubit.measure_in_basis(theta)
                round_results.append(result)
                self.logger.log(f"Servidor {bob_id} mediu o qubit {qubit.qubit_id} na base {theta}, resultado: {result}.")

                # Cliente ajusta o ângulo para o próximo ciclo
                angles[i] = self.adjust_measurement_basis(theta, result)

            measurement_results.append(round_results)
            self.logger.log(f"Resultados da rodada {round_num + 1}: {round_results}")

        self.logger.log(f"Todas as rodadas concluídas. Resultados finais: {measurement_results[-1]}")
        return measurement_results

    def adjust_measurement_basis(self, theta, result):
        """
        Ajusta a base de medição para a próxima rodada, com base no resultado da medição atual.

        Args:
            theta (float): O ângulo de medição atual.
            result (int): Resultado da medição (0 ou 1).

        Returns:
            float: O ângulo ajustado para a próxima rodada de medição.
        """
        delta = 0.1 # Ajuste incremental
        if result == 1:
            return theta + delta # Ajusta para cima se o resultado foi 1
        else:
            return theta - delta # Ajusta para baixo se o resultado foi 0
        
    # MÉTRICAS 
    
    def record_route_fidelities(self, fidelities):
        """
        Registra as fidelidades das rotas na camada de aplicação.

        Args:
            fidelities (list): Lista de valores de fidelidade a serem registrados.
        """
        self.route_fidelities.extend(fidelities)

    def avg_fidelity_on_applicationlayer(self):
        """
        Calcula a média das fidelidades registradas na camada de aplicação.

        Returns:
            float: A média das fidelidades registradas ou 0.0 se a lista estiver vazia.
        """
        # Verifica se há fidelidades registradas.
        if not self.route_fidelities:
            print("Nenhuma fidelidade foi registrada.")
            return 0.0

        # Calcula a média das fidelidades registradas.
        avg_fidelity = sum(self.route_fidelities) / len(self.route_fidelities)
        print(f"A média das fidelidades das rotas é: {avg_fidelity:.4f}")
        return avg_fidelity
    
    def print_route_fidelities(self):
        """
        Imprime a lista de fidelidades das rotas armazenadas.
        """
        if not self.route_fidelities:
            print("Nenhuma fidelidade de rota foi registrada.")
            return

        print("Fidelidades das rotas utilizadas:")
        for fidelity in self.route_fidelities:
            print(f"{fidelity:.4f}")  # Exibe cada fidelidade com 4 casas decimais
            
    def record_used_eprs(self, epr_count):
        """
        Registra o número total de pares EPR usados durante a transmissão.
        
        Args:
            epr_count (int): Total de pares EPR utilizados.
        """
        self.used_eprs += epr_count  # Incrementa o contador de EPRs usados
