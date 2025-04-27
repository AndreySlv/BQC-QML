import time
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit.primitives import Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import Session
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import EfficientSU2, ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import VQC, NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class QuantumMLSimulator:
    def __init__(self, backend=None, log_func=print):
        self.backend = backend or FakeBrisbane()
        self.log = log_func
        self.resultados_vqc = None
        # Dados e configuração
        iris = load_iris()
        X = MinMaxScaler().fit_transform(iris.data)
        y = iris.target
        algorithm_globals.random_seed = 128
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.8, random_state=128)

        self.num_qubits = X.shape[1]
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=1)
        self.ansatz = EfficientSU2(self.num_qubits, reps=1)

    def iniciar_treinamento_vqc(self, max_iter=100):
        self.log(f"Usando backend: {self.backend.name}")
        with Session(backend=self.backend) as session:
            sampler = Sampler()
            objective_vals = []

            def callback(weights, obj_value):
                objective_vals.append(obj_value)
                plt.figure()
                plt.plot(objective_vals)
                plt.title("Objective Function")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.savefig("objective_function_plot.png")
                plt.close()

            optimizer = COBYLA(maxiter=max_iter)
            vqc = VQC(
                sampler=sampler,
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=optimizer,
                callback=callback,
            )

            self.log("Treinando VQC...")
            start_time = time.time()
            vqc.fit(self.X_train, self.y_train)
            duration = round(time.time() - start_time)
            self.log(f"Treino finalizado em {duration} segundos")

            self.resultados_vqc = {
                'accuracy_train': vqc.score(self.X_train, self.y_train),
                'accuracy_test': vqc.score(self.X_test, self.y_test),
                'duration': duration
            }

        from IPython.display import Image, display
        display(Image(filename="objective_function_plot.png"))

    
    def pegar_resultados_vqc(self):
        if self.resultados_vqc is None:
            self.log("Erro: VQC ainda não foi treinado!")
            return None
        return self.resultados_vqc


    def iniciar_qcnn(self, num_images=50, max_iter=100):
        self.log("Iniciando treinamento QCNN...")

        # === Dataset Customizado ===
        def generate_dataset(num_images):
            images, labels = [], []
            hor_array = np.zeros((2, 4))
            ver_array = np.zeros((2, 4))
            hor_array[0][1] = hor_array[0][2] = np.pi / 2
            hor_array[1][0] = hor_array[1][1] = np.pi / 2
            ver_array[0][0] = ver_array[0][2] = np.pi / 2
            ver_array[1][1] = ver_array[1][3] = np.pi / 2

            for n in range(num_images):
                rng = algorithm_globals.random.integers(0, 2)
                if rng == 0:
                    labels.append(0)
                    random_image = algorithm_globals.random.integers(0, 2)
                    images.append(np.array(hor_array[random_image % 2]))
                else:
                    labels.append(1)
                    random_image = algorithm_globals.random.integers(0, 2)
                    images.append(np.array(ver_array[random_image % 2]))
                for i in range(4):
                    if images[-1][i] == 0:
                        images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
            return images, labels

        images, labels = generate_dataset(num_images)
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=246)

        # === Circuitos QCNN ===
        def conv_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            target.cx(1, 0)
            target.rz(np.pi / 2, 0)
            return target

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, qubits)
            return qc

        def pool_circuit(params):
            target = QuantumCircuit(2)
            target.rz(-np.pi / 2, 1)
            target.cx(1, 0)
            target.rz(params[0], 0)
            target.ry(params[1], 1)
            target.cx(0, 1)
            target.ry(params[2], 1)
            return target

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=len(sources) * 3)
            for source, sink in zip(sources, sinks):
                qc = qc.compose(pool_circuit(params[param_index:param_index+3]), [source, sink])
                qc.barrier()
                param_index += 3
            qc_inst = qc.to_instruction()
            qc = QuantumCircuit(num_qubits)
            qc.append(qc_inst, range(num_qubits))
            return qc

        # === Config QCNN ===
        num_qubits = 4
        feature_map = ZFeatureMap(num_qubits)
        ansatz = QuantumCircuit(num_qubits)

        ansatz.compose(conv_layer(4, "c1"), list(range(4)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p1"), list(range(4)), inplace=True)
        ansatz.compose(conv_layer(2, "c2"), list(range(2, 4)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p2"), list(range(2, 4)), inplace=True)

        circuit = QuantumCircuit(num_qubits)
        circuit.compose(feature_map, range(num_qubits), inplace=True)
        circuit.compose(ansatz, range(num_qubits), inplace=True)

        self.log(f"Qubits finais no circuito QCNN: {circuit.num_qubits}")

        sampler = Sampler()
        qnn = SamplerQNN(
            sampler=sampler,
            circuit=circuit,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=lambda x: x % 2,
            output_shape=2
        )

        objective_func_vals = []
        def callback_graph(weights, loss):
            objective_func_vals.append(loss)
            plt.figure()
            plt.plot(objective_func_vals)
            plt.title("QCNN Objective function value against iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective function value")
            plt.savefig("qcnn_objective_function_plot.png")
            plt.close()

        initial_point = np.random.rand(len(ansatz.parameters)).tolist()

        classifier = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=COBYLA(maxiter=max_iter),
            callback=callback_graph,
            initial_point=initial_point
        )

        # === Fit e Avaliação ===
        classifier.fit(np.asarray(train_images), np.asarray(train_labels))
        acc_test = classifier.score(np.asarray(test_images), np.asarray(test_labels))
        acc_train = classifier.score(np.asarray(train_images), np.asarray(train_labels))

        self.resultados_qcnn = {
            'accuracy_train': acc_train,
            'accuracy_test': acc_test,
        }

        self.log(f"Acurácia no teste QCNN: {round(acc_test * 100, 2)}%")
        self.log(f"Acurácia no treino QCNN: {round(acc_train * 100, 2)}%")

        from IPython.display import Image, display
        display(Image(filename="qcnn_objective_function_plot.png"))


    def pegar_resultados_qcnn(self):
        if self.resultados_qcnn is None:
            self.log("Erro: QCNN ainda não foi treinado!")
            return None
        return self.resultados_qcnn
