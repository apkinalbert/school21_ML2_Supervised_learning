import numpy as np

class my_LinearRegression:
    def __init__(self, solver='sgd', learning_rate=0.01, n_epochs=200, random_state=None):
        self.solver = solver
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.random_state = random_state
        self.patience = 7
    
    def _regularization_grad(self, n_samples):
        return 0
    
    def _regularization_loss(self, n_samples):
        return 0
    
    def _compute_loss(self, X, y, n_samples):
        pred = X @ self.weights

        mse =((pred - y) ** 2).mean()

        return mse + self._regularization_loss(n_samples)
    
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
                grad += self._regularization_grad(n_samples)

                self.weights -= self.learning_rate * grad

            loss = self._compute_loss(X, y, n_samples)

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
            grad += self._regularization_grad(n_samples)

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
        else:
            raise ValueError(f'Unknown solver: {self.solver}')
        
        return self
            

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = np.c_[np.ones(X.shape[0]), X]

        return X @ self.weights
    
class my_Ridge(my_LinearRegression):

    def __init__(self, alpha=1.0, solver='sgd', learning_rate=0.01, n_epochs=200, random_state=None):
        super().__init__(solver, learning_rate, n_epochs, random_state)
        self.alpha = alpha

    def _regularization_grad(self, n_samples):
        grad = np.zeros_like(self.weights)
        grad[1:] = 2 * self.alpha * self.weights[1:] / n_samples
        return grad
    
    def _regularization_loss(self, n_samples):
        return self.alpha * np.sum(self.weights[1:] ** 2) / n_samples
    
    def _analytical(self, X, y):
        raise NotImplementedError(
            'Analytical solution is unavailable'
        )
    

class my_Lasso(my_LinearRegression):

    def __init__(self, alpha=1.0, solver='sgd', learning_rate=0.01, n_epochs=200, random_state=None):
        super().__init__(solver, learning_rate, n_epochs, random_state)
        self.alpha = alpha

    def _regularization_grad(self, n_samples):
        grad = np.zeros_like(self.weights)
        grad[1:] = self.alpha * np.sign(self.weights[1:]) / n_samples
        return grad
    
    def _regularization_loss(self, n_samples):
        return self.alpha * np.sum(np.abs(self.weights[1:])) / n_samples
    
    def _analytical(self, X, y):
        raise NotImplementedError(
            'Analytical solution is unavailable'
        )
    
class my_ElasticNet(my_LinearRegression):

    def __init__(self, alpha=1.0, l1_ratio=0.5, solver='sgd', learning_rate=0.01, n_epochs=200, random_state=None):
        super().__init__(solver, learning_rate, n_epochs, random_state)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _regularization_grad(self, n_samples):
        l1 = self.l1_ratio * np.sign(self.weights[1:])
        l2 = 2 * (1 - self.l1_ratio) * self.weights[1:]
        grad = np.zeros_like(self.weights)
        grad[1:] = self.alpha * (l1 + l2) / n_samples

        return grad
    
    def _regularization_loss(self, n_samples):
        l1 = self.l1_ratio * np.sum(np.abs(self.weights[1:]))

        l2 = (1 - self.l1_ratio) * np.sum(self.weights[1:] ** 2)

        return self.alpha * (l1 + l2) / n_samples

    def _analytical(self, X, y):
        raise NotImplementedError(
            'Analytical solution is unavailable'
        )
    

class my_MinMaxScaler:

    def __init__(self, a=0, b=1):
        self.min = None
        self.max = None
        self.a = a
        self.b = b

    def fit(self, X):
        X = np.asarray(X)
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def transform(self, X):
        if self.min is None or self.max is None:
            raise ValueError("Scaler is not fitted")
        
        X = np.asarray(X)
        denominator = np.where(self.max - self.min == 0, 1, self.max - self.min)

        return self.a + (X - self.min) * (self.b - self.a) / denominator
        

class my_StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted")

        X = np.asarray(X)
        std = np.where(self.std == 0, 1, self.std)
        X = (X - self.mean) / std
        return X