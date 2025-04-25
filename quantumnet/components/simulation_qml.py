import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit_ibm_runtime import Sampler,Session
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.circuit.library import EfficientSU2, ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import transpile

class QuantumMLSimulator:
    def __init__(self, backend=None, log_func=print):
        self.backend = backend or FakeBrisbane()
        self.log = log_func
        self.log(f"Simulador iniciado com backend: {self.backend.name}")

        # Carregar dados e preparar dataset
        iris = load_iris()
        X = MinMaxScaler().fit_transform(iris.data)
        y = iris.target
        algorithm_globals.random_seed = 128
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.75, random_state=128)

        self.num_qubits = X.shape[1]
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=1)
        self.ansatz = EfficientSU2(self.num_qubits, reps=1)

    def treinar_vqc(self, max_iter=100):
        with Session(backend=self.backend) as session:
            sampler = Sampler()
            
            transpiled_feature_map = transpile(self.feature_map, backend=self.backend)
            transpiled_ansatz = transpile(self.ansatz, backend=self.backend)

            optimizer = COBYLA(maxiter=max_iter)
            vqc = VQC(
                sampler=sampler,
                feature_map=transpiled_feature_map,
                ansatz=transpiled_ansatz,
                optimizer=optimizer
            )

            self.log("Treinando VQC com o dataset Iris...")
            start = time.time()
            vqc.fit(self.X_train, self.y_train)
            duration = round(time.time() - start)
            self.log(f"Treino finalizado em {duration} segundos")

            accuracy_train = vqc.score(self.X_train, self.y_train)
            accuracy_test = vqc.score(self.X_test, self.y_test)
            self.log(f"Acurácia no treinamento: {accuracy_train}")
            self.log(f"Acurácia no teste: {accuracy_test}")

            return {
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'duration': duration
            }
