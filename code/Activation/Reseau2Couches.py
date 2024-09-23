from Lineaire.Linear import Linear 
from Activation.Tanh import Tanh
from Loss.MSELoss import MSELoss
from Activation.Sigmoide import Sigmoide

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

class Reseau2Couches():
    
    def __init__(self, dim_input, dim_hidden, dim_output=1) -> None:
        """Création du réseau.  
        Args:
            dim_input (_type_): dimension des exemples d'apprentissage
            dim_hidden (_type_): dimension de la sortie de la 1ere couche
            dim_output (int, optional): à 1 par défaut car censée etre
            pour classification binaire 0/1
        """
        
        self.couche_lineaire_1 = Linear(dim_input, dim_hidden)
        self.act_tanh = Tanh()
        self.couche_lineaire_2 = Linear(dim_input, dim_output)
        self.act_sig = Sigmoide()
        self.mse_loss = MSELoss()
        
        self.couts = []
        self.nb_iter = None
        
        self.accuracy_train = None
        self.accuracy_test = None


    def fit(self, datax, datay, nb_iter = 100, learning_rate=10):
        self.nb_iter=nb_iter
        # Passe forward initiale
        output_1 = self.couche_lineaire_1.forward(datax)
        act_1 = self.act_tanh.forward(output_1)
        output_2 = self.couche_lineaire_2.forward(act_1)
        act_2 = self.act_sig.forward(output_2)

        self.couts.append(self.mse_loss.forward(datay, act_2))
        
        for _ in range(nb_iter):
            # passe backward
            ## Calcul des deltas
            delta_1 = self.mse_loss.backward(datay, act_2)
            delta_2 = self.act_sig.backward_delta(output_2, delta_1)
            delta_3 = self.couche_lineaire_2.backward_delta(act_1, delta_2)
            delta_4 = self.act_tanh.backward_delta(output_1, delta_3)
            delta_5 = self.couche_lineaire_1.backward_delta(datax, delta_4)
            
            ## MàJ des gradients
            self.couche_lineaire_2.backward_update_gradient(act_1, delta_2)
            self.couche_lineaire_1.backward_update_gradient(datax, delta_4)

            ## MàJ des paramètres
            self.couche_lineaire_1.update_parameters(gradient_step=learning_rate)
            self.couche_lineaire_2.update_parameters(gradient_step=learning_rate)
            
            #Passe forward
            output_1 = self.couche_lineaire_1.forward(datax)
            act_1 = self.act_tanh.forward(output_1)
            output_2 = self.couche_lineaire_2.forward(act_1)
            act_2 = self.act_sig.forward(output_2)

            self.couts.append(self.mse_loss.forward(datay, act_2))
        else:
            self.accuracy_train = self.accuracy(datay, np.round(act_2))
    
    def plot_couts(self):
        i = list(range(self.nb_iter+1))
        plt.plot(i, self.couts)
    
    def accuracy(self, datay, predict):
        return accuracy_score(datay, predict)
    
    def predict(self, datax):
        output_1 = self.couche_lineaire_1.forward(datax)
        act_1 = self.act_tanh.forward(output_1)
        output_2 = self.couche_lineaire_2.forward(act_1)
        act_2 = self.act_sig.forward(output_2)
    
        return np.round(act_2)
    
    