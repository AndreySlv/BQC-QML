import random
class Epr():
    def __init__(self,  epr_id: int, initial_fidelity: float = None) -> None:
        self._epr_id = epr_id
        self._initial_fidelity = initial_fidelity  if initial_fidelity is not None else random.uniform(0, 1)
        self._current_fidelity = initial_fidelity  if initial_fidelity is not None else random.uniform(0, 1)
    
    @property
    def epr_id(self):
        return self._epr_id
    
    def get_initial_fidelity(self):
        return self._initial_fidelity
    
    def get_current_fidelity(self):
        return self._current_fidelity
    
    def set_fidelity(self, new_fidelity: float):
        """Define a nova fidelidade do par EPR."""
        self._current_fidelity = new_fidelity