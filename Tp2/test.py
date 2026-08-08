import numpy as np
from Part1 import Neuron, Layer, sigmoid, tanh, leaky_ReLu, softmax, MAE, MSE, log_cosh, cross_entropy

n = Neuron()
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.1, 0.2, 0.3])
b = 0.5

print("Neuron:", n(x, w, b, sigmoid))

layer = Layer(input_size=3, output_size=2, input_structure=3)
print("Layer output:", layer(x, sigmoid))

print("sigmoid:", sigmoid(np.array([-1, 0, 1])))
print("tanh:", tanh(np.array([-1, 0, 1])))
print("leaky_ReLu:", leaky_ReLu(np.array([-1, 0, 1])))
print("softmax:", softmax(np.array([2.0, 1.0, 0.1])))

y_true = np.array([0, 1, 0])
y_pred = np.array([0.1, 0.8, 0.1])

print("MAE:", MAE(y_true, y_pred))
print("MSE:", MSE(y_true, y_pred))
print("log_cosh:", log_cosh(y_true, y_pred))
print("cross_entropy:", cross_entropy(y_true, y_pred))