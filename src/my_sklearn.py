import numpy as np

class my_LinearRegression:
    def __init__(self, solver='sgd', learning_rate=0.01, n_epochs=200, random_state=None, batch_size=None):
        self.solver = solver
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.random_state = random_state
        self.patience = 5
        self.batch_size = batch_size
    
    def _regularization_grad(self):
        return 0
    
    def _regularization_loss(self):
        return 0
    
    def _compute_loss(self, X, y, n_samples):
        pred = X @ self.weights

        mse =((pred - y) ** 2).mean()

        return mse + self._regularization_loss() / n_samples
    
    def _gd_family(self, X, y, n_samples, batch_size=1):

        best_weights = self.weights.copy()
        best_loss = np.inf
        bad_epochs = 0

        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.arange(n_samples)

        for _ in range(self.n_epochs):

            np.random.shuffle(indices)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, n_samples, batch_size):

                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = X_batch @ self.weights
                error = y_pred - y_batch

                grad = (2 / len(X_batch)) * X_batch.T @ error
                grad += self._regularization_grad() / len(X_batch)

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
            self._gd_family(X, y, n_samples, batch_size=1)

        elif self.solver == 'gd':
            self._gd_family(X, y, n_samples, batch_size=n_samples)

        elif self.solver == 'mbgd':
            batch_size = self.batch_size or 32
            self._gd_family(X, y, n_samples, batch_size)

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

    def __init__(self, alpha=1.0, solver='sgd', learning_rate=0.01, n_epochs=400, random_state=None):
        super().__init__(solver, learning_rate, n_epochs, random_state)
        self.alpha = alpha

    def _regularization_grad(self):
        grad = np.zeros_like(self.weights)
        grad[1:] = 2 * self.alpha * self.weights[1:]
        return grad
    
    def _regularization_loss(self):
        return self.alpha * np.sum(self.weights[1:] ** 2)
    
    def _analytical(self, X, y):
        I = np.eye(X.shape[1])
        I[0,0] = 0
        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y
    

class my_Lasso(my_LinearRegression):

    def __init__(self, alpha=1.0, solver='sgd', learning_rate=0.01, n_epochs=400, random_state=None):
        super().__init__(solver, learning_rate, n_epochs, random_state)
        self.alpha = alpha

    def _regularization_grad(self):
        grad = np.zeros_like(self.weights)
        grad[1:] = self.alpha * np.sign(self.weights[1:])
        return grad
    
    def _regularization_loss(self):
        return self.alpha * np.sum(np.abs(self.weights[1:]))
    
    def _analytical(self, X, y):
        raise NotImplementedError(
            'Analytical solution is unavailable'
        )
    
class my_ElasticNet(my_LinearRegression):

    def __init__(self, alpha=1.0, l1_ratio=0.5, solver='sgd', learning_rate=0.01, n_epochs=200, random_state=None):
        super().__init__(solver, learning_rate, n_epochs, random_state)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _regularization_grad(self):
        l1 = self.l1_ratio * np.sign(self.weights[1:])
        l2 = 2 * (1 - self.l1_ratio) * self.weights[1:]
        grad = np.zeros_like(self.weights)
        grad[1:] = self.alpha * (l1 + l2)

        return grad
    
    def _regularization_loss(self):
        l1 = self.l1_ratio * np.sum(np.abs(self.weights[1:]))

        l2 = (1 - self.l1_ratio) * np.sum(self.weights[1:] ** 2)

        return self.alpha * (l1 + l2)
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