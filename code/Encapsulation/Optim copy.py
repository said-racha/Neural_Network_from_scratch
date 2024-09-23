import numpy as np
from tqdm import tqdm

from Abstract.Loss import Loss
from Lineaire.Linear import Linear
from Encapsulation.Sequentiel import Sequentiel

from icecream import ic

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
        taille_batch (int): La taille de chaque batch
        loss (Function): La fonction de cout
        nb_iter (int, optional): Nombre d'itérations. Defaults to 100.
        eps (float, optional): Pas de gradient. Defaults to 1e-3.
        shuffle (bool, optional): Si permuter les exemples ou non. Defaults to False.

    Returns:
        optim._couts : La liste des couts calculés par l'optimiseur
        net : Le réseau de neurones entraîné
    """
    ic(X.shape)
    ic(Y.shape)
    #Y = np.reshape(Y, (-1, 1))
    if ((d := len(X.shape) != len(Y.shape)) != 0):  # Dans le cas de la convolution, on a eu une erreurall the input arrays must have same number of dimensions, but the array at index 0 has 3 dimension(s) and the array at index 1 has 2 dimension(s) (X= (5000, 26,1) et Y = (500,10))
                                        #Il faut quand meme verifier la taille de la nouvelle dimension cas ou on a 3 canaux et pas juste 1 
        for i in range(d):
            Y = np.expand_dims(Y, len(Y.shape)+i)
    
    X_Y = np.hstack((X, Y))
    
    if shuffle:
        np.random.shuffle(X_Y)
    
    optim = Optim(net, loss, eps)
    batches = np.array_split(np.array(X_Y), nb_batch)
    for _ in tqdm(range(nb_epochs)):
        
        for batch in batches:
            
            batch_x = np.array([b[:-Y.shape[1]] for b in batch]) 
            batch_y = np.array([b[-Y.shape[1]:] for b in batch])
            
            batch_y = np.squeeze(batch_y)
            
            optim.step(batch_x, batch_y)
        
    
    return optim._net, optim._couts, optim
    