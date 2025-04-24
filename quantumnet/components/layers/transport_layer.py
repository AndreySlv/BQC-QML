import networkx as nx
from quantumnet.components import Host
from quantumnet.objects import Logger, Epr
from random import uniform
import math

class TransportLayer:
    def __init__(self, network, network_layer, link_layer, physical_layer):
        """
        Inicializa a camada de transporte.
        
        args:
            network : Network : Rede.
            network_layer : NetworkLayer : Camada de rede.
            link_layer : LinkLayer : Camada de enlace.
            physical_layer : PhysicalLayer : Camada física.
        """
        self._network = network
        self._physical_layer = physical_layer
        self._network_layer = network_layer
        self._link_layer = link_layer
        self.logger = Logger.get_instance()
        self.transmitted_qubits = []
        self.used_eprs = 0
        self.used_qubits = 0
        self.created_eprs = []  

    def __str__(self):
        """ 
        Retorna a representação em string da camada de transporte. 
        
        returns:
            str : Representação em string da camada de transporte.
        """
        return f'Transport Layer'
    
    def get_used_eprs(self):
        """
        Retorna a lista de pares EPRs usados na camada de transporte.

        Returns:
            list: Lista de pares EPRs usados.
        """
        self.logger.debug(f"Eprs usados na camada {self.__class__.__name__}: {self.used_eprs}")
        return self.used_eprs
    
    def get_used_qubits(self):
        """
        Retorna a lista de qubits usados na camada de transporte.

        Returns:
            list: Lista de qubits usados.
        """
        self.logger.debug(f"Qubits usados na camada {self.__class__.__name__}: {self.used_qubits}")
        return self.used_qubits
    
    def request_transmission(self, alice_id: int, bob_id: int, num_qubits: int):
        """
        Requisição de transmissão de n qubits entre Alice e Bob.
        
        args:
            alice_id : int : Id do host Alice.
            bob_id : int : Id do host Bob.
            num_qubits : int : Número de qubits a serem transmitidos.
            
        returns:
            bool : True se a transmissão foi bem-sucedida, False caso contrário.
        """
        alice = self._network.get_host(alice_id)
        available_qubits = len(alice.memory)

        if available_qubits < num_qubits:
            self.logger.log(f'Número insuficiente de qubits na memória de Alice (Host:{alice_id}). Tentando transmitir os {available_qubits} qubits disponíveis.')
            num_qubits = available_qubits

        if num_qubits == 0:
            self.logger.log(f'Nenhum qubit disponível na memória de Alice ({alice_id}) para transmissão.')
            return False

        max_attempts = 2
        attempts = 0
        success = False

        while attempts < max_attempts and not success:
            self.logger.log(f'Timeslot {self._network.get_timeslot()}: Tentativa de transmissão {attempts + 1} entre {alice_id} e {bob_id}.')
            
            routes = []
            for _ in range(num_qubits):
                route = self._network_layer.short_route_valid(alice_id, bob_id)
                if route is None:
                    self.logger.log(f'Não foi possível encontrar uma rota válida na tentativa {attempts + 1}. Timeslot: {self._network.get_timeslot()}')
                    break
                routes.append(route)
            
            if len(routes) == num_qubits:
                success = True
                for route in routes:
                    for i in range(len(route) - 1):
                        node1 = route[i]
                        node2 = route[i + 1]
                        # Verifica se há pelo menos um par EPR disponível no canal
                        if len(self._network.get_eprs_from_edge(node1, node2)) < 1:
                            self.logger.log(f'Falha ao encontrar par EPR entre {node1} e {node2} na tentativa {attempts + 1}. Timeslot: {self._network.get_timeslot()}')
                            success = False
                            break
                    if not success:
                        break
            
            if not success:
                attempts += 1

        if success:
            # Registrar os qubits transmitidos
            for route in routes:
                qubit_info = {
                    'route': route,
                    'alice_id': alice_id,
                    'bob_id': bob_id,
                }
                self.transmitted_qubits.append(qubit_info)
            self.logger.log(f'Transmissão de {num_qubits} qubits entre {alice_id} e {bob_id} concluída com sucesso. Timeslot: {self._network.get_timeslot()}')
            return True
        else:
            self.logger.log(f'Falha na transmissão de {num_qubits} qubits entre {alice_id} e {bob_id} após {attempts} tentativas. Timeslot: {self._network.get_timeslot()}')
            return False

    def teleportation_protocol(self, alice_id: int, bob_id: int):
        """
        Realiza o protocolo de teletransporte de um qubit de Alice para Bob.
        
        args:
            alice_id : int : Id do host Alice.
            bob_id : int : Id do host Bob.
        
        returns:
            bool : True se o teletransporte foi bem-sucedido, False caso contrário.
        """
        
        # Estabelece uma rota válida
        route = self._network_layer.short_route_valid(alice_id, bob_id)
        if route is None:
            self.logger.log(f'Não foi possível encontrar uma rota válida para teletransporte entre {alice_id} e {bob_id}. Timeslot: {self._network.get_timeslot()}')
            return False
        
        # Pega um qubit de Alice e um qubit de Bob
        alice = self._network.get_host(alice_id)
        bob = self._network.get_host(bob_id)
        
        if len(alice.memory) < 1 or len(bob.memory) < 1:
            self.logger.log(f'Alice ou Bob não possuem qubits suficientes para teletransporte. Timeslot: {self._network.get_timeslot()}')
            return False
        
        qubit_alice = alice.memory.pop(0)  
        qubit_bob = bob.memory.pop()       
        
        # Calcula a fidelidade final do teletransporte
        f_alice = qubit_alice.get_current_fidelity()
        f_bob = qubit_bob.get_current_fidelity()
        
        # Assume fidelidade do link como a média das fidelidades dos pares EPR na rota
        fidelities = []
        for i in range(len(route) - 1):
            epr_pairs = self._network.get_eprs_from_edge(route[i], route[i+1])
            fidelities.extend([epr.get_current_fidelity() for epr in epr_pairs])
        
        if not fidelities:
            self.logger.log(f'Não foi possível encontrar pares EPR na rota entre {alice_id} e {bob_id}. Timeslot: {self._network.get_timeslot()}')
            return False
        
        f_route = sum(fidelities) / len(fidelities)
        
        # Fidelidade final do qubit teletransportado
        F_final = f_alice * f_bob * f_route + (1 - f_alice) * (1 - f_bob) * (1 - f_route)
        
        qubit_info = {
            'alice_id': alice_id,
            'bob_id': bob_id,
            'route': route,
            'fidelity_alice': f_alice,
            'fidelity_bob': f_bob,
            'fidelity_route': f_route,
            'F_final': F_final,
            'qubit_alice': qubit_alice,
            'qubit_bob': qubit_bob,
            'success': True
        }
        
        # Adiciona o qubit teletransportado à memória de Bob com a fidelidade final calculada
        qubit_alice.fidelity = F_final
        bob.memory.append(qubit_alice)
        self.logger.log(f'Teletransporte de qubit de {alice_id} para {bob_id} foi bem-sucedido com fidelidade final de {F_final}. Timeslot: {self._network.get_timeslot()}')
        
        # Par virtual é deletado no final
        for i in range(len(route) - 1):
            self._network.remove_epr(route[i], route[i + 1])
        
        self.transmitted_qubits.append(qubit_info)
        return True

    def avg_fidelity_on_transportlayer(self):
        """
        Calcula a fidelidade média de todos os qubits realmente utilizados na camada de transporte.

        returns:
            float : Fidelidade média dos qubits utilizados na camada de transporte.
        """
        total_fidelity = 0
        total_qubits_used = 0

        # Calcula a fidelidade dos qubits transmitidos e registrados no log de qubits transmitidos
        for qubit_info in self.transmitted_qubits:
            fidelity = qubit_info['F_final']
            total_fidelity += fidelity
            total_qubits_used += 1
            self.logger.log(f'Fidelidade do qubit utilizado de {qubit_info["alice_id"]} para {qubit_info["bob_id"]}: {fidelity}')

        # Considera apenas os qubits efetivamente transmitidos (não inclui os qubits que permanecem na memória dos hosts)
        if total_qubits_used == 0:
            self.logger.log('Nenhum qubit foi utilizado na camada de transporte.')
            return 0.0

        avg_fidelity = total_fidelity / total_qubits_used
        self.logger.log(f'A fidelidade média de todos os qubits utilizados na camada de transporte é {avg_fidelity}')
        
        return avg_fidelity


    def get_teleported_qubits(self):
        """
        Retorna a lista de qubits teletransportados.
        
        returns:
            list : Lista de dicionários contendo informações dos qubits teletransportados.
        """
        return self.transmitted_qubits
    
    def run_transport_layer(self, alice_id: int, bob_id: int, num_qubits: int, route=None):
        """
        Executa a requisição de transmissão e o protocolo de teletransporte.

        args:
            alice_id : int : Id do host Alice.
            bob_id : int : Id do host Bob.
            num_qubits : int : Número de qubits a serem transmitidos.

        returns:
            bool : True se a operação foi bem-sucedida, False caso contrário.
        """
        alice = self._network.get_host(alice_id)
        bob = self._network.get_host(bob_id)
        available_qubits = len(alice.memory)

        # Se Alice tiver menos qubits do que o necessário, crie mais qubits
        if available_qubits < num_qubits:
            qubits_needed = num_qubits - available_qubits
            for _ in range(qubits_needed):
                # self._network.timeslot()
                self._physical_layer.create_qubit(alice_id, increment_timeslot = False)
            available_qubits = len(alice.memory)

        if available_qubits != num_qubits:
            self.logger.log(f'Erro: Alice tem {available_qubits} qubits, mas deveria ter {num_qubits} qubits. Abortando transmissão.')
            return False

        max_attempts = 2
        attempts = 0
        success_count = 0
        route_fidelities = []  
        used_eprs = 0 

        while attempts < max_attempts and success_count < num_qubits:
            for _ in range(num_qubits - success_count):
                # Usa a rota fornecida ou calcula uma nova rota, se necessário
                if route is None:
                    route = self._network_layer.short_route_valid(alice_id, bob_id)
                    if route is None:
                        self.logger.log(f'Não foi possível encontrar uma rota válida na tentativa {attempts + 1}.')
                        break
                else:
                    self.logger.log(f"Usando a rota fornecida: {route}")


                # Verifica a fidelidade dos pares EPR ao longo da rota
                fidelities = []
                eprs_used_in_current_transmission = 0  # Contador de EPRs para a rota atual
                for i in range(len(route) - 1):
                    node1 = route[i]
                    node2 = route[i + 1]
                    epr_pairs = self._network.get_eprs_from_edge(node1, node2)
                    
                    # Seleciona apenas os pares EPR necessários para a transmissão de um qubit
                    if epr_pairs:
                        fidelities.append(epr_pairs[0].get_current_fidelity())
                        eprs_used_in_current_transmission += 1
                    else:
                        self.logger.log(f'Não foi possível encontrar pares EPR suficientes na rota {route[i]} -> {route[i + 1]}.')
                        break
            
                if not fidelities:
                    attempts += 1
                    continue

                f_route = sum(fidelities) / len(fidelities)
                
                if alice.memory:
                    #self._network.timeslot()
                    qubit_alice = alice.memory.pop(0)
                    f_alice = qubit_alice.get_current_fidelity()
                    
                    F_final = f_alice * f_route
                    route_fidelities.append(F_final) 

                    qubit_alice.fidelity = F_final
                    bob.memory.append(qubit_alice)

                    success_count += 1
                    self.used_qubits += 1
                    used_eprs += eprs_used_in_current_transmission 
                    self.logger.log(f'Timeslot {self._network.get_timeslot()}: Teletransporte de qubit de {alice_id} para {bob_id} na rota {route} foi bem-sucedido com fidelidade final de {F_final}.')

                    self.transmitted_qubits.append({
                        'alice_id': alice_id,
                        'bob_id': bob_id,
                        'route': route,
                        'fidelity_alice': f_alice,
                        'fidelity_route': f_route,
                        'F_final': F_final,
                        'timeslot': self._network.get_timeslot(),
                        'qubit': qubit_alice
                    })
                else:
                    self.logger.log(f'Alice não possui qubits suficientes para continuar a transmissão.')
                    break

            attempts += 1

        # Passa a lista de fidelidades finais para a camada de aplicação
        self._network.application_layer.record_route_fidelities(route_fidelities)
        self._network.application_layer.record_used_eprs(used_eprs)  # Registra apenas EPRs usados na transmissão bem-sucedida

        if success_count == num_qubits:
            self.logger.log(f'Transmissão e teletransporte de {num_qubits} qubits entre {alice_id} e {bob_id} concluídos com sucesso.')
            return True
        else:
            self.logger.log(f'Falha na transmissão de {num_qubits} qubits entre {alice_id} e {bob_id}. Apenas {success_count} qubits foram transmitidos com sucesso.')
            return False

    def run_transport_layer_eprs(self, alice_id: int, bob_id: int, num_qubits: int, route=None, is_return=False, scenario=1):
        """
        Executa a requisição de transmissão e o protocolo de teletransporte para protocolo Andrews Childs.

        args:
            alice_id : int : Id do host Alice.
            bob_id : int : Id do host Bob.
            num_qubits : int : Número de qubits a serem transmitidos.
            route : list : Rota a ser usada (opcional).
            is_return : bool : Indica se é a etapa de retorno (evitar recriação de EPRs no cenário 1).
            scenario : int : Define o cenário de simulação (1 ou 2).

        returns:
            bool : True se a operação foi bem-sucedida, False caso contrário.
        """
        alice = self._network.get_host(alice_id)
        bob = self._network.get_host(bob_id)
        available_qubits = len(alice.memory)

        # Garantir qubits suficientes
        if available_qubits < num_qubits:
            qubits_needed = num_qubits - available_qubits
            for _ in range(qubits_needed):
                self._physical_layer.create_qubit(alice_id, increment_timeslot=False)
            available_qubits = len(alice.memory)

        if available_qubits != num_qubits:
            self.logger.log(f'Erro: Alice tem {available_qubits} qubits, mas deveria ter {num_qubits} qubits. Abortando transmissão.')
            return False

        # Calcular rota, se necessário
        if route is None:
            route = self._network_layer.short_route_valid(alice_id, bob_id, increment_timeslot=False)
            if route is None:
                self.logger.log('Não foi possível encontrar uma rota válida.')
                return False
        else:
            self.logger.log(f"Usando a rota fornecida: {route}")
        
        # Lógica para Gerar Pares EPRs Baseada no Cenário
        if scenario == 1:
            if not is_return:
                # Criar todos os pares EPRs no início
                num_eprs_per_channel = num_qubits * 2
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    for _ in range(num_eprs_per_channel):
                        epr_pair = self._physical_layer.create_epr_pair(fidelity=1.0, increment_timeslot=False, increment_eprs=False)
                        self._physical_layer.add_epr_to_channel(epr_pair, (u, v))
                self.logger.log(f'{num_eprs_per_channel} pares EPRs criados para cada segmento da rota {route}.')
            else:
                self.logger.log(f"Etapa de retorno: consumindo EPRs existentes na rota {route}.")
        if scenario == 2:
            # Dividir os pares EPRs entre ida e volta
            eprs_to_create = (num_qubits * 2) // 2
            if not is_return:
                # Criar metade dos pares EPRs na ida
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    self.logger.log(f"Ida: Criando {eprs_to_create} pares EPRs no segmento {u} -> {v}.")
                    for _ in range(eprs_to_create):
                        epr_pair = self._physical_layer.create_epr_pair(fidelity=1.0, increment_timeslot=False, increment_eprs=False)
                        self._physical_layer.add_epr_to_channel(epr_pair, (u, v))
            elif is_return:
                # Criar a outra metade na volta
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    self.logger.log(f"Volta: Criando {eprs_to_create} pares EPRs no segmento {u} -> {v}.")
                    for _ in range(eprs_to_create):
                        epr_pair = self._physical_layer.create_epr_pair(fidelity=1.0, increment_timeslot=False, increment_eprs=True)
                        self._physical_layer.add_epr_to_channel(epr_pair, (u, v))

        success_count = 0
        total_eprs_used = 0
        route_fidelities = []

        while success_count < num_qubits:
            # Verificar fidelidade da rota
            f_route = self.calculate_average_fidelity(route)
            self.logger.log(f"Fidelidade atual da rota: {f_route}")

            # Consumir EPRs da rota
            eprs_used_in_current_transmission = 0
            for i in range(len(route) - 1):
                node1 = route[i]
                node2 = route[i + 1]
                epr_pairs = self._network.get_eprs_from_edge(node1, node2)
                if len(epr_pairs) == 0:
                    self.logger.log(f"Sem pares EPRs disponíveis no segmento {node1} -> {node2}. Interrompendo transmissão.")
                    return False

                # Usar o primeiro EPR disponível
                epr_pair = epr_pairs[0]
                eprs_used_in_current_transmission += 1
                total_eprs_used += 1
                self._network.remove_epr(node1, node2)

            self._network.timeslot()
            # Teletransportar qubit
            if alice.memory:
                qubit_alice = alice.memory.pop(0)
                f_qubit = qubit_alice.get_current_fidelity()
                F_final = f_route
                qubit_alice.fidelity = F_final
                bob.memory.append(qubit_alice)

                self.logger.log(f"Fidelidade final: {F_final:.4f} (F_qubit: {f_qubit:.4f} * F_rota: {f_route:.4f})")
                if F_final < 0.85:
                    self.logger.log(f"Fidelidade final {F_final:.4f} abaixo de 0.85. Interrompendo transmissão.")
                    break

                success_count += 1
                route_fidelities.append(F_final)

        # Registros e finalização
        self._network.application_layer.record_route_fidelities(route_fidelities)
        self._network.application_layer.record_used_eprs(total_eprs_used)
        self.logger.log(f"Foram utilizados {total_eprs_used} pares EPRs ao longo da transmissão.")
        
        if success_count == num_qubits:
            self.logger.log(f'Transmissão de {num_qubits} qubits entre {alice_id} e {bob_id} concluída com sucesso.')
            return True
        else:
            self.logger.log(f'Transmissão falhou. Apenas {success_count} qubits foram transmitidos com sucesso.')
            self.register_failed_request(alice_id, bob_id, num_qubits, route, "Transmissão incompleta")
            return False


    def run_transport_layer_eprs_bfk(self, alice_id: int, bob_id: int, num_qubits: int, route=None, is_return=False, scenario=1):
        """
        Executa a requisição de transmissão e o protocolo de teletransporte para o protocolo BFK.

        Args:
            alice_id : int : Id do host Alice.
            bob_id : int : Id do host Bob.
            num_qubits : int : Número de qubits a serem transmitidos.
            route : list : Rota a ser usada (opcional).
            is_return : bool : Indica se é a etapa de retorno.
            scenario : int : Define o cenário de simulação (1 ou 2).

        Returns:
            bool : True se a operação foi bem-sucedida, False caso contrário.
        """
        alice = self._network.get_host(alice_id)
        bob = self._network.get_host(bob_id)

        # Garantir qubits suficientes
        if len(alice.memory) < num_qubits:
            qubits_needed = num_qubits - len(alice.memory)
            for _ in range(qubits_needed):
                self._physical_layer.create_qubit(alice_id, increment_timeslot=False)

        if len(alice.memory) != num_qubits:
            self.logger.log(f"Timeslot {self._network.get_timeslot()} Erro: Alice tem {len(alice.memory)} qubits, mas deveria ter {num_qubits} qubits. Abortando transmissão.")
            return False

        # Calcular rota, se necessário
        if route is None:
            route = self._network_layer.short_route_valid(alice_id, bob_id, increment_timeslot=False)
            if route is None:
                self.logger.log(f"Timeslot {self._network.get_timeslot()} Não foi possível encontrar uma rota válida.")
                return False
        else:
            self.logger.log(f"Timeslot {self._network.get_timeslot()} Usando a rota fornecida: {route}")

        success_count = 0
        total_eprs_used = 0
        fidelidades_finais = []

        # Cenário 2: Criar todos os pares EPRs no início
        if scenario == 2 and not is_return:
            self.logger.log(f"Timeslot {self._network.get_timeslot()} Iniciando criação de pares EPRs para o Cenário 2.")
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                for _ in range(num_qubits):
                    epr_pair = self._physical_layer.create_epr_pair(fidelity=1.0, increment_timeslot=False)
                    self._physical_layer.add_epr_to_channel(epr_pair, (u, v))
                self.logger.log(f"Timeslot {self._network.get_timeslot()} Pares EPRs criados para o enlace {u} -> {v}.")
            self.logger.log(f"Timeslot {self._network.get_timeslot()} Pares EPRs criados para toda a rota.")

        while success_count < num_qubits:
            # Cenário 1: Criar EPRs a cada transmissão
            if scenario == 1:
                self.logger.log(f"Timeslot {self._network.get_timeslot()} Iniciando criação de pares EPRs para o Cenário 1.")
                for i in range(len(route) - 1):
                    u, v = route[i], route[i + 1]
                    epr_pair = self._physical_layer.create_epr_pair(fidelity=1.0, increment_timeslot=False)
                    self._physical_layer.add_epr_to_channel(epr_pair, (u, v))
                    self.logger.log(f"Timeslot {self._network.get_timeslot()} Par EPR criado e adicionado ao canal {u} -> {v}. Avançando timeslot...")
                    self._network.timeslot()

            # Consumir EPRs da rota
            for i in range(len(route) - 1):
                node1 = route[i]
                node2 = route[i + 1]
                epr_pairs = self._network.get_eprs_from_edge(node1, node2)
                if not epr_pairs:
                    self.logger.log(f"Timeslot {self._network.get_timeslot()} Sem pares EPRs disponíveis no segmento {node1} -> {node2}. Interrompendo transmissão.")
                    return False

                # Usar o primeiro EPR disponível e logar fidelidade
                epr_pair = epr_pairs[0]
                fidelity_epr = epr_pair.get_current_fidelity()
                self.logger.log(f"Timeslot {self._network.get_timeslot()} EPR consumido no segmento {node1} -> {node2} com fidelidade {fidelity_epr:.4f}.")
                total_eprs_used += 1
                self._network.remove_epr(node1, node2)

            # Teletransportar um único qubit
            if alice.memory:
                self._network.timeslot()
                qubit_alice = alice.memory.pop(0)
                f_final = qubit_alice.get_current_fidelity()
                qubit_alice.fidelity = f_final
                bob.memory.append(qubit_alice)

                self.logger.log(f"Timeslot {self._network.get_timeslot()} Fidelidade final do teletransporte: {f_final:.4f}")
                
                # Verificar fidelidade final do qubit
                if f_final < 0.85:
                    self.logger.log(f"Timeslot {self._network.get_timeslot()} Fidelidade final {f_final:.2f} abaixo de 0.85. Interrompendo transmissão.")
                    break
                
                success_count += 1
                fidelidades_finais.append(f_final)

        # Registros e finalização
        self._network.application_layer.record_route_fidelities(fidelidades_finais)
        self._network.application_layer.record_used_eprs(total_eprs_used)
        self.logger.log(f"Timeslot {self._network.get_timeslot()} Foram utilizados {total_eprs_used} pares EPRs ao longo da transmissão.")

        # Log final dos EPRs restantes na rota
        self.logger.log("Pares EPRs restantes na rota:")
        for i in range(len(route) - 1):
            node1 = route[i]
            node2 = route[i + 1]
            remaining_eprs = len(self._network.get_eprs_from_edge(node1, node2))
            self.logger.log(f"Timeslot {self._network.get_timeslot()} Segmento {node1} -> {node2}: {remaining_eprs} pares EPRs restantes.")

        if success_count == num_qubits:
            self.logger.log(f"Timeslot {self._network.get_timeslot()} Transmissão de {num_qubits} qubits entre {alice_id} e {bob_id} concluída com sucesso.")
            return True
        else:
            self.logger.log(f"Timeslot {self._network.get_timeslot()} Transmissão falhou. Apenas {success_count} qubits foram transmitidos com sucesso.")
            self.register_failed_request(alice_id, bob_id, num_qubits, route, "Transmissão incompleta")
            return False

    def clear_eprs_from_route(self, route: list):
        """
        Remove todos os pares EPRs restantes dos canais em uma rota.

        Args:
            route (list): Lista de nós representando a rota.
        """
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            channel = (u, v)
            self._physical_layer.remove_all_eprs_from_channel(channel)
            self.logger.log(f"Todos os pares EPRs removidos do canal {u} -> {v}.")


    def register_failed_request(self, alice_id, bob_id, num_qubits, route, reason):
        """
        Registra uma requisição que falhou diretamente no controlador.

        Args:
            alice_id (int): ID de Alice.
            bob_id (int): ID de Bob.
            num_qubits (int): Número de qubits envolvidos.
            route (list): Rota utilizada ou None se nenhuma rota foi encontrada.
        """
        failed_request = {
            "alice_id": alice_id,
            "bob_id": bob_id,
            "num_qubits": num_qubits,
            "route": route,
        }
        if hasattr(self._network, 'controller') and self._network.controller:
            self._network.controller.record_failed_request(failed_request)
        self.logger.log(f"Falha registrada: {failed_request}")

    def calculate_average_fidelity(self, route):
        """
        Calcula a fidelidade média ao longo de uma rota especificada na rede.

        Args:
            route (list): Lista de nós que compõem a rota (ex: [u, v, w]).

        Returns:
            float: O produto das fidelidades dos pares EPR ao longo da rota, ou 0.0 se algum 
            canal não tiver pares disponíveis.
        """
        fidelities = []

        # Percorre a rota, considerando pares de nós consecutivos como canais.
        for i in range(len(route)-1):
            u, v = route[i], route[i+1]
            # Obtém os pares EPR do canal atual.
            eprs = self._network.get_eprs_from_edge(u, v)
            if not eprs:
                self.logger.log(f"Sem pares EPR disponíveis no canal {u}->{v}. Fidelidade = 0.")
                return 0.0
            # Obtém a fidelidade do último EPR disponível no canal.
            fidelity = eprs[-1].get_current_fidelity()  
            self.logger.log(f"Fidelidade do EPR {u}->{v}: {fidelity}")
            fidelities.append(fidelity)

        # Se existirem fidelidades, calcula o produto delas.
        if fidelities:
            product = 1.0
            for f in fidelities:
                product *= f
            self.logger.log(f"Produto das fidelidades para rota {route}: {product}")
            return product
        return 0.0

