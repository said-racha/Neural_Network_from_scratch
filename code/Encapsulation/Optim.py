import numpy as np
from tqdm import tqdm

from Abstract.Loss import Loss
from Lineaire.Linear import Linear
from Encapsulation.Sequentiel import Sequentiel


class Optim():
    
    def __init__(self, net:Sequentiel|Linear, loss:Loss, eps:float):
        """Optimiseur qui effectue une étape d'optimisation des paramètres pour le 
        réseau de neurones (calcule la loss et son gradient et mes à jour les paramètres)

        Args:
            net (Module): le réseau de neurone (module ou sequentiel)
            loss (function): la fonction cout
            eps (float): Pas de gradient
            
        Attributs supplémentaires:
            _cout = la liste des valeurs du cout
        """
        
        self._net:Sequentiel|Linear = net
        self._net.zero_grad()
        self._loss:Loss = loss 
        self._eps:float = eps
        self._couts:list[float] = []
    
    def step(self, batch_x:np.ndarray, batch_y:np.ndarray):
        output:np.ndarray = self._net.forward(batch_x) 
        cout:float = self._loss.forward(batch_y, output)
        gradient_loss:np.ndarray = self._loss.backward(batch_y, output)
        
        self._net.backward(batch_x, gradient_loss)
        self._net.update_parameters(self._eps)
        self._couts.append(cout.mean())
        
        
    
    def score(self, Y, pred):
        return np.where(Y == pred, 1, 0).mean()
    

def SGD(net, X:np.ndarray, Y:np.ndarray, nb_batch:int, loss:Loss, nb_epochs=10, eps:float=1e-5, shuffle:bool=False):
    """Effectue la descente de gradient stochastique/batch.

    Args:
        net (Module): Le réseau de neurone ou le module
        X (np.ndarray): L'ensemble des exemples de train
        Y (np.ndarray): L'ensemble des labels de train
        nb_batch (int): Le nombre de batchs
        loss (Function): La fonction de cout
        nb_epochs (int, optional): Nombre d'itérations. Defaults to 100.
        eps (float, optional): Pas de gradient. Defaults to 1e-3.
        shuffle (bool, optional): Si permuter les exemples ou non. Defaults to False.

    Returns:
        optim._couts : La liste des couts calculés par l'optimiseur
        optim.net : Le réseau de neurones entraîné
        optim : l'optimiseur
    """
    
    indices = np.arange(X.shape[0])

    if shuffle :
        np.random.shuffle(indices)

    #Séparer les indices en "nb_batch" sous-ensembles
    batches_indices = np.array_split(indices, nb_batch)
    
    
    optim = Optim(net, loss, eps)
    
    for _ in tqdm(range(nb_epochs)):
        
        for batch in batches_indices:
            optim.step(X[batch], Y[batch])
        
    return optim._net, optim._couts, optim
    