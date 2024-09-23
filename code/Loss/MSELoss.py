import numpy as np
from Abstract.Loss import *

class MSELoss(Loss):
    
    def __init__(self):
        return
    
    def forward(self, y:np.ndarray, yhat:np.ndarray) -> float:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        assert y.shape == yhat.shape

        return np.sum((y-yhat)**2, axis=1)
    
    def backward(self, y:np.ndarray, yhat:np.ndarray) -> np.ndarray:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        assert y.shape == yhat.shape
        return 2*(yhat-y) #De taille (n,k)
    
    