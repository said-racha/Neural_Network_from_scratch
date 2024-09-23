from Abstract.Module import Module
import numpy as np

class ReLU(Module):
    def __init__(self, threshold=0, name='ReLU'):
        super().__init__()
        self._name = name
        self._threshold = threshold

    def zero_grad(self):
        pass

    def forward(self, X):
        return np.where(X>self._threshold, X, 0)
        
    def derivee(self,input):
        return (input > self._threshold).astype(float)

    def backward_update_gradient(self, input, delta) -> None:
        pass
    
    def update_parameters(self, gradient_step=1e-3) -> None:
        pass

    def backward_delta(self, input, delta):
        return np.multiply(delta, self.derivee(input))
        

