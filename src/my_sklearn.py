import numpy as np

class my_LinearRegression:
    def __init__(self, solver='sgd', learning_rate=0.01, n_epochs=200, random_state=None):
        self.solver = solver
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.random_state = random_state
        self.patience = 7

    def _compute_loss(self, X, y):
        pred = X @ self.weights
        return ((pred - y) ** 2).mean()
    
    def _sgd(self, X, y, n_samples):

        best_weights = self.weights.copy()
        best_loss = np.inf
        bad_epochs = 0

        if self.random_state is not None:
            np.random.seed(self.random_state)

        for _ in range(self.n_epochs):

            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in indices:
                xi = X[i]
                yi = y[i]

                y_pred = xi @ self.weights

                error = y_pred - yi
                grad = 2 * error * xi

                self.weights -= self.learning_rate * grad

            loss = self._compute_loss(X, y)

            if loss < best_loss:
                best_loss = loss
                best_weights = self.weights.copy()
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= self.patience:
                break
        
        self.weights = best_weights

    def _gd(self, X, y, n_samples):

        best_weights = self.weights.copy()
        best_loss = np.inf
        bad_epochs = 0
    
        for _ in range(self.n_epochs):

            y_pred = X @ self.weights
            error = y_pred - y

            grad = (2 / n_samples) * X.T @ error

            self.weights -= self.learning_rate * grad

            loss = self._compute_loss(X, y)

            if loss < best_loss:
                best_loss = loss
                best_weights = self.weights.copy()
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= self.patience:
                break
        
        self.weights = best_weights

    def _analytical(self, X, y):
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        X = np.c_[np.ones(X.shape[0]), X]

        n_features = X.shape[1]
        n_samples = X.shape[0]
        self.weights = np.zeros(n_features)

        if self.solver == 'sgd':
            self._sgd(X, y, n_samples)
        elif self.solver == 'gd':
            self._gd(X, y, n_samples)
        elif self.solver == 'analytical':
            self._analytical(X, y)
            

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = np.c_[np.ones(X.shape[0]), X]

        return X @ self.weights
    
