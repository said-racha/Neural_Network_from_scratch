from Abstract.Module import Module
import numpy as np

class SoftMax(Module):
    
    def __init__(self, name='Softmax'):
        super().__init__()
        self._name = name
        
    def forward(self, input:np.ndarray) -> float:

        e = np.exp(input)
        return e / np.sum(e, axis=1).reshape((-1, 1))

    def zero_grad(self):
        pass
    
    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input:np.ndarray, delta:np.ndarray):
        pass
    
    def backward_delta(self, input:np.ndarray, delta:np.ndarray):
        softmax = self.forward(input)
        
        return delta * (softmax * (1 - softmax))
        
        
        
        
        
    