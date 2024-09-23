import numpy as np
from Abstract.Loss import *

class BCELoss(Loss):
    
    def __init__(self, name='BCELoss'):
        super().__init__()
        self._name = name
        
    def forward(self, y:np.ndarray, yhat:np.ndarray) -> float:
        
        # y_hat_truey + sum(exp(y_hat_i)) sur toutes les classes
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        assert y.shape == yhat.shape
        
        
        return - np.mean(y*np.log(np.clip(yhat, 1e-8, np.max(yhat))) + (1-y)*np.log(np.clip(1-yhat, 1e-8, np.max(1-yhat))), axis=1)
        

    def zero_grad(self):
        pass
    
    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input:np.ndarray, delta:np.ndarray):
        pass
    
    def backward(self, y:np.ndarray, yhat:np.ndarray) -> np.ndarray:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        assert y.shape == yhat.shape
        
        yhat = np.clip(yhat, 1e-10, 1)
        
        return -(y / yhat - (1 - y) / np.clip(1 - yhat, 1e-10, 1))
        
