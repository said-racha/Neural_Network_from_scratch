import numpy as np
from Abstract.Loss import Loss

class CELogSoftMax(Loss):
    
    def __init__(self, name='CELogSoftMax'):
        super().__init__()
        self._name = name
        
    def forward(self, y:np.ndarray, yhat:np.ndarray) -> float:
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        try:
            assert y.shape == yhat.shape
        except AssertionError as e:
            print(f'Les dimensiosn de y {y.shape} et yhat {yhat.shape} sont incompatibles')
        eps = 1e-8  # Pour éviter le log de zero
        yhat_truey = np.sum(yhat*y, axis=1)
        
        s = np.log(np.sum(np.exp(yhat))+eps)
        
        return -yhat_truey + s

    def zero_grad(self):
        pass
    
    def backward(self, y:np.ndarray, yhat:np.ndarray) -> np.ndarray:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        assert y.shape == yhat.shape
        
        yhat_exp = np.exp(yhat)
        
        return yhat_exp / np.sum(yhat_exp, axis=1).reshape((-1,1)) - y
        
        
        
        
    