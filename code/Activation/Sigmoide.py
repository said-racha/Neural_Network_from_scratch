from Abstract.Module import Module
import numpy as np

class Sigmoide(Module):
    
    def __init__(self, name='sig'):
        super().__init__()
        self._name = name
        
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def zero_grad(self):
        pass
    
    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input:np.ndarray, delta:np.ndarray):
        pass
    
    def backward_delta(self, input:np.ndarray, delta:np.ndarray):
        ## Calcul la derivee de l'erreur p.r aux entrées (sorties du module précédent)
        # input=Z_h-1 car la dérivée se fait par rapport aux Z de la couche precédente
        # Ca va être le delta qu'on va transmettre à la couche précédente
        # delta de la forme output*dim_loss (1 pour l'instant)
        
        assert delta.shape == input.shape
        return delta * (self.forward(input)*(1-self.forward(input)))
    
    