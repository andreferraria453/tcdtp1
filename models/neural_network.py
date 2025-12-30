import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(10,), learning_rate_init=0.01, epochs=100, random_state=None, verbose=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        
        # Atributos internos
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        
        # --- MUDANÇA 1: Variáveis estilo Sklearn ---
        self.loss_ = 0.0          # Guarda o loss da última iteração
        self.loss_curve_ = []     # (Opcional) Guarda o histórico todo se quiseres usar depois
        
        self.is_initialized = False
        self.encoder = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _initialize_weights(self, n_features, n_classes):
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_hidden = self.hidden_layer_sizes[0]
        
        self.weights1 = np.random.randn(n_features, n_hidden) * 0.1
        self.bias1 = np.zeros((1, n_hidden))
        self.weights2 = np.random.randn(n_hidden, n_classes) * 0.1
        self.bias2 = np.zeros((1, n_classes))
        
        self.is_initialized = True

    def partial_fit(self, X, y, classes=None):
        """Treina uma única iteração e guarda o loss em self.loss_"""
        if classes is None:
            classes = np.unique(y)
        
        if self.encoder is None:
            self.encoder = OneHotEncoder(categories=[classes], sparse_output=False, dtype=np.float64, handle_unknown='ignore')
            self.encoder.fit(classes.reshape(-1, 1))

        n_features = X.shape[1]
        n_classes = len(classes)
        
        if not self.is_initialized:
            self._initialize_weights(n_features, n_classes)

        y_encoded = self.encoder.transform(y.reshape(-1, 1))

        # --- Forward ---
        z1 = np.dot(X, self.weights1) + self.bias1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self._softmax(z2)

        # --- Backward ---
        m = X.shape[0]
        dz2 = a2 - y_encoded
        dw2 = (1 / m) * np.dot(a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self._sigmoid_derivative(a1)
        dw1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # --- Update ---
        self.weights1 -= self.learning_rate_init * dw1
        self.bias1    -= self.learning_rate_init * db1
        self.weights2 -= self.learning_rate_init * dw2
        self.bias2    -= self.learning_rate_init * db2

        # --- MUDANÇA 2: Guardar Loss e Retornar self ---
        loss_value = -np.mean(np.sum(y_encoded * np.log(a2 + 1e-8), axis=1))
        
        self.loss_ = loss_value          # Atribui ao estilo sklearn
        self.loss_curve_.append(loss_value) 
        
        return self  # Retorna o objeto, permitindo encadeamento (chaining)

    def fit(self, X, y):
        self.loss_curve_ = []
        classes = np.unique(y)
        self.encoder = None
        self.is_initialized = False
        
        for i in range(self.epochs):
            self.partial_fit(X, y, classes=classes)
            
            if self.verbose and (i == 0 or (i + 1) % 10 == 0 or i == self.epochs - 1):
                print(f"Epoch {i+1}/{self.epochs} - Loss: {self.loss_:.5f}")
                    
        return self

    def predict(self, X):
        z1 = np.dot(X, self.weights1) + self.bias1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self._softmax(z2)
        indices = np.argmax(a2, axis=1)
        return self.encoder.categories_[0][indices]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))