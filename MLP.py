import numpy as np

#np.random.seed(42) #NEW, best on seed=2

'''
TODO:

commentler silinecek / chatgpt gibi olmayacak
optimizer eklenecek
RQ bulunacak

dropout?
batch normalization?
l1? l2?
early stopping?
learning rate decay?
momentum?
weight initialization?
'''

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step for Adam

        # Xavier initialization
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

        # Initialize Adam parameters
        self.m_w = [np.zeros_like(w) for w in self.weights]  # First moment estimate for weights
        self.v_w = [np.zeros_like(w) for w in self.weights]  # Second moment estimate for weights
        self.m_b = [np.zeros_like(b) for b in self.biases]  # First moment estimate for biases
        self.v_b = [np.zeros_like(b) for b in self.biases]  # Second moment estimate for biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -100, 100))) # changed from -500, 500 -k (prevents overflow)
    
    def sigmoid_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        for w, b in zip(self.weights, self.biases):
            X = self.sigmoid(X.dot(w) + b)
            self.activations.append(X)
        return X
    
    def backward(self, X, y, output):
        """Backward pass using Adam optimizer"""
        self.t += 1  # Increase time step

        deltas = [(y - output) * self.sigmoid_derivative(output)]
        for i in range(len(self.weights) - 1, 0, -1):
            deltas.append(deltas[-1].dot(self.weights[i].T) * self.sigmoid_derivative(self.activations[i]))
        deltas.reverse()

        for i in range(len(self.weights)):
            grad_w = self.activations[i].T.dot(deltas[i])  # Gradient of weights
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)  # Gradient of biases

            # Update first moment estimates (momentum)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b

            # Update second moment estimates (RMSprop-like scaling)
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b ** 2)

            # Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Adam update rule
            self.weights[i] += (self.learning_rate * m_w_hat) / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] += (self.learning_rate * m_b_hat) / (np.sqrt(v_b_hat) + self.epsilon)
            
    def train(self, X, y, epochs=100, batch_size=32):
        """Train the network with mini-batch gradient descent"""
        best_accuracy = 0
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
            
        # Calculate and print metrics
            metrics = self.evaluate(X, y)
            loss = np.mean(np.square(y - self.forward(X))) # MSE LOSS (mean squared error)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return (output > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """
        Evaluate the model's performance using accuracy, precision, recall, and F1-score.
        """
        predictions = self.predict(X).flatten()
        y = y.flatten()
        
        TP = np.sum((predictions == 1) & (y == 1))
        FP = np.sum((predictions == 1) & (y == 0))
        TN = np.sum((predictions == 0) & (y == 0))
        FN = np.sum((predictions == 0) & (y == 1))
        
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
    
def preprocess_data(data, train_ratio=0.8):
    """Preprocess mushroom data and split into train/test sets"""
    #STALK ROOT HAS BEEN DELETED AS A FEATURE AS IT CONTAINS MISSING VALUES
    encoding_dicts = {
        'class': {'e': 0, 'p': 1},
        'cap-shape': {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5},
        'cap-surface': {'f': 0, 'g': 1, 'y': 2, 's': 3},
        'cap-color': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9},
        'bruises': {'t': 0, 'f': 1},
        'odor': {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8},
        'gill-attachment': {'a': 0, 'd': 1, 'f': 2, 'n': 3},
        'gill-spacing': {'c': 0, 'w': 1, 'd': 2},
        'gill-size': {'b': 0, 'n': 1},
        'gill-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11},
        'stalk-shape': {'e': 0, 't': 1},
        'stalk-surface-above-ring': {'f': 0, 'y': 1, 'k': 2, 's': 3},
        'stalk-surface-below-ring': {'f': 0, 'y': 1, 'k': 2, 's': 3},
        'stalk-color-above-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
        'stalk-color-below-ring': {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
        'veil-type': {'p': 0, 'u': 1},
        'veil-color': {'n': 0, 'o': 1, 'w': 2, 'y': 3},
        'ring-number': {'n': 0, 'o': 1, 't': 2},
        'ring-type': {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7},
        'spore-print-color': {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8},
        'population': {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
        'habitat': {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6}
    }
    
    X_data = []
    y_data = []
    
    for line in data.strip().split('\n'):
        features = line.strip().split(',')
        y_data.append(1 if features[0] == 'p' else 0)
        
        encoded_features = []
        for i, feature in enumerate(features[1:]):
            feature_key = list(encoding_dicts.keys())[i]
            one_hot = [0] * len(encoding_dicts[feature_key])
            index = encoding_dicts[feature_key].get(feature, -1)
            if index != -1:
                one_hot[index] = 1
            encoded_features.extend(one_hot)
        
        X_data.append(encoded_features)
    
    X = np.array(X_data, dtype=float)
    y = np.array(y_data, dtype=float).reshape(-1, 1)
    
    # Normalize data
    #X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    
    # Split into training and testing sets
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)

    # Apply the normalization
    X_train = (X_train - mean_train) / (std_train + 1e-8)
    X_test = (X_test - mean_train) / (std_train + 1e-8)  # Use training mean/std
    
    return X_train, y_train, X_test, y_test, X, y

def run_mushroom_classification(data, hidden_size=[8], learning_rate=0.01, epochs=10, batch_size=32):
    """Run the mushroom classification model with batch training"""
    X_train, y_train, X_test, y_test, _, _ = preprocess_data(data)
    input_size = X_train.shape[1]
    output_size = 1  # Binary classification
    
    model = MLP(input_size, hidden_size, output_size, learning_rate)
    print("Training model...")
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Set Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return model

def try_different_neuron_amounts():
    neurons = []
    for x in range(1, 11):
        for y in range(1, 5):
            if y == 3:
                continue
            temp_list = [x] * y
            neurons.append(temp_list)
    return neurons

def try_different_hyperparams():
    with open('processed.data', 'r') as f:
        mushroom_data = f.read()

    X_train, y_train, X_test, y_test, _, _ = preprocess_data(mushroom_data)

    input_size = X_train.shape[1]  # Ensure input_size matches the dataset
    hidden_sizes = try_different_neuron_amounts()

    all_metrics = []
    third = 0
    for x in hidden_sizes: #removed epoch loop
            if x == [1]:
                third = 0
            if x == [3, 3, 3, 3]:
                third = 1
            if x == [6, 6, 6, 6]:
                third = 2
            if x == [8, 8, 8, 8]:
                third = 3
            for z in [16, 32, 64]:  # Batch sizes, removed 8 and 128 batch size
                for w in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:  # Learning rates, removed 0.05
                    model_x_layers = MLP(input_size=input_size, hidden_sizes=x, output_size=1, learning_rate=w)
                    print(f"\nTraining model with {x} hidden layers, batch size {z}, learning rate {w}...")
                    model_x_layers.train(X_train, y_train, epochs=100, batch_size=z)

                    metrics_x_layers = model_x_layers.evaluate(X_test, y_test)
                    all_metrics.append((metrics_x_layers, x, z, w))

                    match third:
                        case 0:
                            print('STILL TRAINING')
                            print('STILL TRAINING')
                            print('STILL TRAINING')
                        case 1:
                            print('%33')
                            print('%33')
                            print('%33')
                        case 2:
                            print('%66')
                            print('%66')
                            print('%66')
                        case 3:
                            print('CLIMAXING')
                            print('CLIMAXING')
                            print('CLIMAXING')

    # Print results after training
    print("\nFinal Results:")
    with open('results.txt', 'w') as f:
        all_metrics_sorted = sorted(all_metrics, key=lambda x: x[0]['accuracy'], reverse=True)
        for metric in all_metrics_sorted:
            print(f"ACC: {metric[0]} | HL: {(metric[1])} | BS: {metric[2]} | LR: {metric[3]}")
            f.write(f"ACC: {metric[0]} | HL: {(metric[1])} | BS: {metric[2]} | LR: {metric[3]}\n")

if __name__ == "__main__":
    try_different_hyperparams()
    '''
    with open('processed.data', 'r') as f:
        mushroom_data = f.read()
        run_mushroom_classification(mushroom_data, hidden_size=[8, 8], learning_rate=0.01, epochs=100, batch_size=32)
    '''