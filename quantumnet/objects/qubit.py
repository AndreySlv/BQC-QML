import random
import math

class Qubit():
    def __init__(self, qubit_id: int, initial_fidelity: float = None) -> None:
        self.qubit_id = qubit_id
        self._qubit_state = 0  
        self._phase = 1  
        self._initial_fidelity = initial_fidelity if initial_fidelity is not None else random.uniform(0.9, 1)
        self._current_fidelity = self._initial_fidelity

    def __str__(self):
        return f"Qubit {self.qubit_id} with state {self._qubit_state} and phase {self._phase}"

    def update_fidelity(self):
        self._current_fidelity = random.uniform(0, 1)

    def get_initial_fidelity(self):
        return self._initial_fidelity

    def get_current_fidelity(self):
        return self._current_fidelity

    def set_current_fidelity(self, new_fidelity: float):
        """
        Define a fidelidade atual do qubit.
        """
        self._current_fidelity = new_fidelity

    def apply_x(self):
        """
        Aplica a porta X (NOT) ao qubit.
        """
        self._qubit_state = 1 if self._qubit_state == 0 else 0

    def apply_y(self):
        """Aplica a porta Y ao qubit."""
        # Porta Y transforma |0> em i|1> e |1> em -i|0>.
        self._qubit_state = 1 if self._qubit_state == 0 else 0
        self._phase *= -1  # Representa a rotação de fase imaginária

    def apply_z(self):
        """Aplica a porta Z ao qubit."""
        # Porta Z transforma |0> em |0> e |1> em -|1>.
        if self._qubit_state == 1:
            self._phase *= -1  # Representa a inversão de fase do estado |1>

    def apply_hadamard(self):
        """Aplica a porta Hadamard (H) ao qubit."""
        # Hadamard transforma o estado |0> em (|0> + |1>) / sqrt(2)
        # e |1> em (|0> - |1>) / sqrt(2). Para simulação, usa-se probabilidade.
        if self._qubit_state == 0:
            self._qubit_state = random.choice([0, 1])  
        else:
            self._qubit_state = random.choice([0, 1])  
            
        # Alteração de fase com 50% de chance simula o comportamento quântico
        self._phase = random.choice([1, -1])

    def measure(self):
        """
        Realiza a medição do qubit no estado atual.
        """
        return self._qubit_state


    def apply_controlled_phase(self, control_qubit):
        """
        Aplica a operação de fase controlada (C-phase) ao qubit atual condicionalmente ao estado do control_qubit.

        Args:
            control_qubit (Qubit): O qubit de controle da operação C-phase.
        """
        if control_qubit._qubit_state == 1:  # Se o qubit de controle estiver no estado |1>
            self.apply_z()  # Aplica a porta Z no qubit alvo (modifica a fase)

    def measure_in_basis(self, theta):
        """
        Mede o qubit na base definida pelo ângulo theta.

        Args:
            theta (float): O ângulo da base de medição (em radianos).

        Returns:
            int: O resultado da medição (0 ou 1).
        """
        # Simula a aplicação de uma rotação em função do ângulo theta.
        # O cálculo usa uma aproximação simples de base de medição.
        prob_0 = (1 + math.cos(theta)) / 2  # Probabilidade de medir |0>
        
        # Realiza a medição com base na probabilidade calculada.
        result = 0 if random.uniform(0, 1) < prob_0 else 1
        return result