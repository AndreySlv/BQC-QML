#INSTALE AS DEPENDENCIAS 
# pip install qiskit qiskit-aer numpy matplotlib
#wget https://qulacs-quantum-datasets.s3.us-west-1.amazonaws.com/base_test_mnist_784_f90.zip
#unzip base_test_mnist_784_f90.zip
import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt

path = "base_test_mnist_784_f90/qasm/"

def show_figure(pict: np.array, index=0):
    plt.imshow(pict.reshape(28, 28), cmap="Greys")
    plt.axis('off')
    plt.savefig(f"figura_{index}.png")
    plt.show()

def show_state_figure(state, index=0):
    pict = state
    result = [abs(pict[x]) for x in range(28 * 28)]
    show_figure(np.array(result), index=index)

simulator = AerSimulator(method='statevector')


print("Arquivos encontrados:", os.listdir(path))

for i in range(5):
    file_path = path + str(i) + ".qasm"  # ajuste a extensão conforme necessário
    if not os.path.exists(file_path):
        print(f"Arquivo {file_path} não encontrado.")
        continue

    with open(file_path) as f:
        qasm = f.read()
        qc = QuantumCircuit.from_qasm_str(qasm)
        
        qc.save_statevector()
        
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit)
        result = job.result()
        
        state = result.data(0)['statevector'] 
        show_state_figure(state, index=i)