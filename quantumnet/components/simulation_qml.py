import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit_ibm_runtime import Sampler, Session
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.circuit.library import EfficientSU2, ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit import transpile

class QuantumMLSimulator:
    def __init__(self, backend=None, log_func=print, max_samples=24):
        self.backend = backend or FakeBrisbane()
        self.log = log_func
        self.resultados_vqc = None
        # Dados e configuração
        iris = load_iris()
        X = MinMaxScaler().fit_transform(iris.data)
        y = iris.target
        algorithm_globals.random_seed = 128
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=128)

        # Limitar as amostras
        self.X_train = X_train[:max_samples]
        self.y_train = y_train[:max_samples]
        self.X_test = X_test[:max_samples]
        self.y_test = y_test[:max_samples]

        self.num_qubits = X.shape[1]
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=1)
        self.ansatz = EfficientSU2(self.num_qubits, reps=1)

    def iniciar_treinamento_vqc(self, max_iter=100):
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

            self.log("Treinando VQC...")
            start_time = time.time()
            vqc.fit(self.X_train, self.y_train)
            duration = round(time.time() - start_time)

            self.vqc_resultados = {
                'accuracy_train': vqc.score(self.X_train, self.y_train),
                'accuracy_test': vqc.score(self.X_test, self.y_test),
                'duration': duration
            }
            self.log(f"Treino concluído: {self.vqc_resultados}")

    def pegar_resultados_vqc(self):
        if self.vqc_resultados is None:
            self.log("Erro: VQC ainda não foi treinado!")
            return None
        return self.vqc_resultados
