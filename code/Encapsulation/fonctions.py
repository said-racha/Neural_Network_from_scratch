import numpy as np
import matplotlib.pyplot as plt

from utils import tools

def evaluate_encapsulation(opt, datax, datay, testx, testy) :
    pred_train = np.where(opt._net.forward(datax)>=0.5,1,0)
    pred_test = np.where(opt._net.forward(testx)>=0.5,1,0)
    
    print("accuracy train: ", opt.score(datay, pred_train))
    print("accuracy test: ", opt.score(testy, pred_test))

    fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(20,5))
    ax.flatten()

    tools.plot_frontiere(datax,opt._net.predict,ax=ax[0])
    tools.plot_data(datax, datay,ax[0])
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Frontière de décison en apprentissage")

    tools.plot_frontiere(testx, opt._net.predict, ax=ax[1])
    tools.plot_data(testx, testy, ax[1])
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("x2")
    ax[1].set_title(" frontiere de decison en test")


    ax[2].plot(np.arange(len(opt._couts)), opt._couts, color='red')
    ax[2].legend(["Cout"])
    ax[2].set_xlabel("Nombre d'epochs")
    ax[2].set_ylabel("MSE")
    ax[2].set_title("Variation de la MSE")
    plt.show()

def evaluate_bceloss(opt, X_train, X_test, y_train, y_test):
    plt.plot(np.arange(len(opt._couts)), opt._couts)
    plt.title("Evolution du cout")
    plt.xlabel("Cout CELogSoftMax")
    plt.ylabel("Nombre d'itérations")
    plt.show()

    raw_scores_train = opt._net.forward(X_train)
    raw_scores_test = opt._net.forward(X_test)

    print("accuracy train: ", score(y_train, raw_scores_train))
    print("accuracy test: ", score(y_test, raw_scores_test))
    
    return raw_scores_train, raw_scores_test


def predict(opt, i, X, Y):
    """Compresse puis reconstruit l'image à l'indice i
    Args:
        opt (_type_): L'optimiseur
        i (int): L'indice de l'image à prédire
        X (np.ndarray): Features
        Y (np.ndarray): labels
    """
    dim = int(np.sqrt(X.shape[1]))
    
    plt.figure(figsize=(5, 2))  
    
    plt.subplot(1, 2, 1)  # Première cellule de la grille
    plt.imshow(X[i].reshape((dim,dim)), cmap='grey')
    plt.title(f'Image originale ({Y[i]})')
    pred = opt._net.forward(X[i].reshape((1,dim**2)))
    plt.subplot(1, 2, 2)  # Deuxième cellule de la grille
    plt.imshow(pred.reshape((dim,dim)), cmap='grey')
    plt.title('Image reconstruite')
    plt.show()
    #print('classe = ', y_train[i])
    
def evaluate(opt, X, Y):
    """ Affiche la reconstruction pour les 10 premieres images du dataset. et plot le cout
    
    Args:
        opt (Optim): l'Optimiseur
        X (np.ndarray): images
        Y (np.ndarray): labels
    """
    for i in range(10):
        predict(opt, i, X, Y)

    plt.plot(range(len(opt._couts)), opt._couts)
    plt.show()
    
######################################

def afficher_images_cluster_rep(groupe_df_cluster, X, Y):
    plt.figure(figsize=(10, 4))

    for i, image in enumerate(groupe_df_cluster.RepresentationLatente.head(10)):
        
        plt.subplot(2, 5, i+1)
        
        
        plt.imshow(X[image].reshape(16, 16), cmap='gray')
        plt.title(Y[image])
        plt.axis('off')
        
    # Ajuster les marges
    plt.tight_layout()
    plt.show()
    
def afficher_images_cluster(groupe_df_cluster, X, Y):
    plt.figure(figsize=(10, 4))

    for i, image in enumerate(groupe_df_cluster.Image.head(10)):
        
        plt.subplot(2, 5, i+1)
        
        plt.imshow(X[image].reshape(16, 16), cmap='gray')
        plt.title(Y[image])
        plt.axis('off')
        
    # Ajuster les marges
    plt.tight_layout()
    plt.show()

######################################


def transform_one_hot(y):
    """Transforme l'ensemble des labels en vecteurs one-hot

    Args:
        Y (np.ndarray): L'ensemble des labels

    Returns:
        Matrice des labels one-hot
    """
    one_hot_vectors = np.zeros((y.shape[0], np.max(y)+1))
    one_hot_vectors[np.arange(y.shape[0]), y] = 1
    
    return one_hot_vectors

def pred_classes(y_hat):
    """Retourne les classes à prédire à partir des probabilités

    Args:
        y_hat : Vecteurs one-hot

    Returns:
        Vecteurs de classes (int)
    """
    # Juste un reshape pour avoir une matrice avec une ligne si jamais on a un vecteur des probas d'un seul exemple
    if len(y_hat.shape) == 1:
        y_hat = y_hat.reshape((1,-1))
    
    return  np.argmax(y_hat, axis=1)
    

def score(y, yhat):
    """_summary_
    Args:
        y (np.ndarray[int]):Vecteur de classes
        yhat (np.ndarray[float]): Vecteurs des probas de chaque classe
    Returns:
        Accuracy 
    """
    assert (len(y.shape) == 1) or (y.shape[1]==1) , 'La supervision doit etre un vecteur de classes (entiers)'
    assert (len(yhat.shape) == 2) , 'Les prédictions doivent être les vecteurs de probabilités des classes pour chaque exemples'
    
    predictions = pred_classes(yhat)
    nb_bonnes_reponses = np.sum(np.where((y-predictions)==0, 1, 0))

    return  nb_bonnes_reponses/len(predictions) #s, s/len(yhat)

