import numpy as np
import math

class LinearRegression:
    def __init__(self, x_train, y_train):
        '''
            Intialise the LinearRegression Object
        
            Args:
                x_train (ndarray (m,n): training data, m examples with n features
                y_train (ndarray (m,)): target values
                iterations (scalar): number of iterations, default value is 100000
                w (ndarray (n,)): model parameters  
                b (scalar)      : model parameter
            Returns:
                self: LinearRegression class object
        '''
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self):
        """
            Computes the y_prediction for linear regression 
        
            Returns:
                y_pred (ndarray (m,)): prediction data based on the model parameters w and b
        """
        y_pred = np.dot(self.x_train, self.w) + self.b
        return y_pred
    
    def __compute_cost(self):
        """
            Computes the cost for linear regression
            
            Returns:
                cost (scalar)       : The cost of the current model with model parameter w and b
        """
        m, n = self.x.shape
        y_pred = self.predict()
        cost = (y_pred - self.y_train)**2
        total_cost = np.sum(cost) / 2 / m
                
        return total_cost
    
    def __compute_gradients(self):
        """
            Computes the gradients (n, ), n features, for linear regression
            
            Returns:
                dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
                dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
        """
    
        m, n = self.x_train.shape
        dj_dw = np.zeros((n, ))
        dj_db = 0.

        for i in range(m):
            for j in range(n):
                dj_dw[j] += (self.predict() - self.y_train[i]) * self.x_train[i][j]
            dj_db += self.predict() - self.y_train[i]

        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db
    
    def fit(self, w_init, b_init, alpha, iterations):
        """
            Performs batch gradient descent
            
            Args:
                w_init (ndarray (n,)): Initial values of model parameters  
                b_init (scalar)      : Initial values of model parameter
                alpha (float)      : Learning rate
                iterations (scalar) : number of iterations to run gradient descent
            
            Returns:
                w (ndarray (n,))   : Updated values of parameters
                b (scalar)         : Updated value of parameter 
        """
        
        self.w = w_init
        self.b = b_init
        
        self.J_history = []
        
        for i in range(iterations):
            dj_dw, dj_db = self.__compute_gradients()
            self.w = self.w - alpha * dj_dw
            self.b = self.b - alpha * dj_db

            if i < 100000:
                self.J_history.append(self.__compute_cost())

            if (i % math.ceil(iterations/10)) == 0:
                print(f"Iteration {i:4d}: Cost {self.J_history[-1]:8.2f}")
                