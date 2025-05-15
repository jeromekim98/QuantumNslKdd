from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.primitives import Sampler

def get_quantum_kernel(num_features):
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
    kernel = QuantumKernel(feature_map=feature_map, sampler=Sampler())
    return kernel