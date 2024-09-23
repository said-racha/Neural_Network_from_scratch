import numpy as np
from pandas import DataFrame

from Abstract.Module import *

class Linear(Module):
    
    def __init__(self, input_dim:int, output_dim:int, name:str='Linear', init_type=0):
        """_summary_

        Args:
            input (int): nombre d'entrées
            output (int): nombre de sorties
        
        Penser à eut etre ajouter le biais
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._name = name
        
        if init_type == 'random':
            self._parameters = np.random.randn(input_dim, output_dim)*10
        else:
            self._parameters = self.params(input_dim, output_dim, init_type)
        
        self._biais = self.params(1, output_dim, init_type)
        
        self._gradient = np.zeros((input_dim, output_dim))  #Car le gradient est de la forme nb_entrée*nb_sorties, et la loss a 1 seule sortie
        self._gradient_biais = np.zeros((1, output_dim))
        
        
    def params(self, n,d,dat):
        np.random.seed(5)
        if dat == 1:
            parameters = np.random.normal(0, 1, (n, d)) * np.sqrt(2 / (n + d))
        elif dat == 0:
            parameters = np.random.normal(0, 1, (n, d))
        else :
            parameters = np.random.normal(0, 1, (n, d)) -0.5
        return parameters

    
    def reset(self):
        self._parameters = np.random.randn(self._input_dim, self._output_dim)*10
        self.zero_grad()
    
    def forward(self, X:np.ndarray) -> np.ndarray:
        """Calcule la sortie à partir de X

        Args:
            X (:np.ndarray): Les entrées du module
        """
        #input_dim = X.shape[1]
        try:
            assert X.shape[1] == self._input_dim
        except AssertionError as e:
            print(f'Le nombre de neurones de {self._name} est incompatible avec la dimension des entrées ({X.shape[1]})'.format(e))
            
        return X @ self._parameters + self._biais
    
    def zero_grad(self) -> None:
        self._gradient = np.zeros(self._gradient.shape)
        self._gradient_biais = np.zeros(self._gradient_biais.shape)
    
    def update_parameters(self, gradient_step=1e-3) -> None:
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient
        self._biais -= gradient_step*self._gradient_biais
        

    def backward_update_gradient(self, input:np.ndarray, delta:np.ndarray) -> None:
        ## 
        """Calcule le gradient de la loss p.r aux paramètres
            et Met a jour la valeur du gradient

        Args:
            input (_type_): Les entrée du module
            delta ( np.array((nb_sorties_couche_courante,)) ): 
        """
        # la dérivée des sorties du module p.r aux parametres est l'input du module
        # La somme sur les k sorties se fait dans le produit matriciel
        # input=X car la dérivée se fait par rapport aux paramètres W
        
        assert input.shape[1] == self._input_dim
        assert delta.shape[1] == self._output_dim # == self._parameters.shape[1] 
        assert delta.shape[0] == input.shape[0] 
        
        try:
            self._gradient += input.T @ delta 
            self._gradient_biais += np.sum(delta, axis=0)
            
        except ValueError as e:
            print(str(e))
            print(f"Erreur dans {self._name}")

    def backward_delta(self, input:np.ndarray, delta:np.ndarray) -> np.ndarray:
        ## Calcul la derivee de l'erreur p.r aux entrées (sorties du module précédent)
        # input=Z_h-1 car la dérivée se fait par rapport aux Z de la couche precédente
        # Ca va être le delta qu'on va transmettre à la couche précédente
        # delta de la forme output*dim_loss (1 pour l'instant)
        
        assert input.shape[1] == self._input_dim
        assert delta.shape[1] == self._output_dim # == self._parameters.shape[1] 
        assert delta.shape[0] == input.shape[0]
        
        return delta @ self._parameters.T
    
    def describe_shape(self):
            d = {'name':[self._name] , 'input_dim':[self._input_dim], 'output_dim':[self._output_dim]}
            
            df = DataFrame(d)
            print(df)
    
    def describe_values(self):
            d = {'name':[self._name] , 'parametres': [self._parameters], 'gradient':[self._gradient]}
            
            df = DataFrame(d)
            print(df)