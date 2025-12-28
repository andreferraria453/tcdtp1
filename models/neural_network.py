import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(50,), learning_rate_init=0.01, epochs=100, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.epochs = epochs
        self.random_state = random_state
        
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        self.loss_ = [] 
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
        """
        Treina a rede neuronal em um lote (batch) ou época individual.
        Lida com strings ou números nas labels.
        """
        # 1. Garantir que as classes são um array do NumPy
        if classes is None:
            classes = np.unique(y)
        else:
            classes = np.asarray(classes)

        # 2. Configuração do OneHotEncoder (Evita o erro de TypeError: isnan)
        # Definimos dtype=object para suportar strings e evitar cálculos matemáticos em texto
        if not hasattr(self, 'encoder') or self.encoder is None:
            self.encoder = OneHotEncoder(
                categories=[classes], 
                sparse_output=False, 
                dtype=object,
                handle_unknown='ignore'
            )
            self.encoder.fit(classes.reshape(-1, 1))

        # 3. Inicialização de Pesos (Apenas na primeira chamada)
        n_features = X.shape[1]
        n_classes = len(classes)
        if not hasattr(self, 'weights1') or self.weights1 is None:
            self._initialize_weights(n_features, n_classes)

        # 4. Transformar as etiquetas (y) em One-Hot Encoding
        # O reshape(-1, 1) é obrigatório para o transform do sklearn
        y_encoded = self.encoder.transform(y.reshape(-1, 1))

        # -----------------------------------------------------------
        # A. FORWARD PROPAGATION
        # -----------------------------------------------------------
        # Camada Oculta
        z1 = np.dot(X, self.weights1) + self.bias1
        a1 = self._activation_function(z1)  # ex: sigmoid ou relu

        # Camada de Saída
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self._softmax(z2)             # Predições (probabilidades)

        # -----------------------------------------------------------
        # B. BACKPROPAGATION
        # -----------------------------------------------------------
        m = X.shape[0] # número de amostras no lote

        # Erro na saída
        dz2 = a2 - y_encoded
        dw2 = (1 / m) * np.dot(a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Erro na camada oculta
        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self._activation_derivative(z1)
        dw1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # -----------------------------------------------------------
        # C. ATUALIZAÇÃO DOS PESOS (Gradiente Descendente)
        # -----------------------------------------------------------
        self.weights1 -= self.learning_rate * dw1
        self.bias1    -= self.learning_rate * db1
        self.weights2 -= self.learning_rate * dw2
        self.bias2    -= self.learning_rate * db2

        # Opcional: Calcular e guardar a perda desta iteração
        self.loss = -np.mean(np.sum(y_encoded * np.log(a2 + 1e-8), axis=1))
        
        return self

    def fit(self, X, y):
        """Fit padrão (treina todas as épocas de uma vez)"""
        self.loss_history = []
        classes = np.unique(y)
        
        for _ in range(self.epochs):
            self.partial_fit(X, y, classes=classes)
            self.loss_history.append(self.loss_)
        return self

    def predict(self, X):
        # Forward pass simples para obter a classe final
        z1 = np.dot(X, self.weights1) + self.bias1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = self._softmax(z2)
        
        # Retorna o índice da maior probabilidade convertido para a classe original
        indices = np.argmax(a2, axis=1)
        return self.encoder.categories_[0][indices]

    def score(self, X, y):
        # Necessário para a tua função _train_eval_model calcular acc no histórico
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))