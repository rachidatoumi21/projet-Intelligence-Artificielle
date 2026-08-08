import numpy as np
import torch

# Partie de réseaux neuronaux
class Neuron():
    def __init__(self):
        pass

    def __call__(self, x, w, b, activation) -> float:
        x = np.array(x, dtype=float)
        w = np.array(w, dtype=float)
        b = float(b)

        z = np.sum(x * w) + b
        return activation(z)


class Layer():
    def __init__(self, input_size, output_size, input_structure):
        # TODO initialiser la matrice de poids et le vecteur de biais avec des valeurs appropriées
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(output_size, input_size) * 0.01

        self.biases = np.zeros(output_size, dtype=float)

        
        if input_structure is None:
            self.input_structure = np.ones((output_size, input_size), dtype=float)
        else:
            mask = np.array(input_structure, dtype=float)

            if mask.shape == (input_size, output_size):
                mask = mask.T

            self.input_structure = mask.reshape(output_size, input_size)

        
        self.neurons = [Neuron() for _ in range(output_size)]

    def __call__(self, x, activation):
        # TODO calculez le vecteur de sortie de la couche
        x = np.array(x, dtype=float).reshape(-1)

        masked_weights = self.weights * self.input_structure

        outputs = []
        for i in range(self.output_size):
            y = self.neurons[i](x, masked_weights[i], self.biases[i], activation)
            outputs.append(y)

        return np.array(outputs, dtype=float)

# Fonctions d'activation
def sigmoid(x):
    x = np.array(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    x = np.array(x, dtype=float)
    return np.tanh(x)

def leaky_ReLu(x, alpha=0.01):
    x = np.array(x, dtype=float)
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    x = np.array(x, dtype=float)
    x_stable = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x_stable)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# Fonctions de perte
def MAE(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def MSE(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)

def log_cosh(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    diff = y_pred - y_true
    return np.mean(np.log(np.cosh(diff)))

def cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    losses = -np.sum(y_true * np.log(y_pred + epsilon), axis=-1)
    return np.mean(losses)


def train(model, epochs, optimizer, criterion, train_loader, test_loader, device):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    print("Starting training...")
    for epoch in range(epochs):

        # Entrainement
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Évaluation
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = running_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f} | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

    return train_losses, test_losses, train_accuracies, test_accuracies