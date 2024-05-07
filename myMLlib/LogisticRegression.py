import numpy as np

class LogisticRegression:
    def __init__():
        print('Hello World')
    
    def sigmoid_function(self, z):
        '''
            Compute the sigmoid function, with z
        
            Args:
                z (scalar): the y_pred = w * x + b
            Returns:
                sigmoid: value of sigmoid
        '''

        return 1 / (1 + np.exp(-z))