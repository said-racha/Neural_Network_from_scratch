from abc import ABC, abstractmethod
from numpy import ndarray

class Loss(ABC):
    
    @abstractmethod
    def forward(self, y, yhat) -> float:
        pass

    @abstractmethod
    def backward(self, y, yhat) -> ndarray:
        pass