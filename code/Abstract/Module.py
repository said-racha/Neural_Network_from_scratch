from abc import ABC, abstractmethod
from numpy import ndarray

class Module(ABC):
    def __init__(self):
        self._input_dim = None
        self._output_dim = None
        self._name = 'name'
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    @abstractmethod
    def forward(self, X:ndarray) -> ndarray:
        ## Calcule la passe forward
        pass

    @abstractmethod
    def update_parameters(self, gradient_step=1e-3) -> None:
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        #self._parameters -= gradient_step*self._gradient
        pass

    @abstractmethod
    def backward_update_gradient(self, input, delta) -> None:
        ## Met a jour la valeur du gradient
        pass

    @abstractmethod
    def backward_delta(self, input, delta) -> ndarray:
        ## Calcul la derivee de l'erreur
        pass
