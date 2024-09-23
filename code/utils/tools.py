#Fichier du TME de MAPSI

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
# from keras.datasets import mnist

def plot_data(data,labels=None,ax=plt):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    if labels is not None:
        labels = labels.reshape(-1)
    cols,marks = ["red", "green", "blue", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        ax.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        ax.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

def plot_frontiere(data,f,step=20,ax=plt):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    ax.contourf(x,y,f(grid).reshape(x.shape),colors=('gray','blue'),levels=[-1,0,1])

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//2)
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//2)
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y.reshape(-1, 1)

###########################################################################################################################

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def params(n,d,dat):

   if dat == 1:
     parameters = np.random.normal(0, 1, (n, d)) * np.sqrt(2 / (n + d))
   elif dat == 0:
     parameters = np.random.normal(0, 1, (n, d))
   else :
     parameters = np.random.normal(0, 1, (n, d)) -0.5
   return parameters

###########################################################################################################################


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
    

def print_image(X,Y,net,size,n):
    fig,ax=plt.subplots(nrows=np.math.ceil( size/4 ),ncols=4,figsize=(20,5))
    ax=ax.flatten()

    choice=np.random.choice(np.arange(X.shape[0]),size=size)
    fig.tight_layout()

    for pos,i in enumerate(choice) :
        y_hat= pred_classes(net.predict(np.array([X[i]]))) #np.where(net.predict(np.array([X[i]])) == 1)[0].item()  # a la base y_hat=net.predict(np.array([X[i]])
        ax[pos].imshow(X[i].reshape((n,n)),interpolation="nearest",cmap="gray")
        ax[pos].set_title(f'classe predit {y_hat[0]} / vrai class {Y[i]}')
        ax[pos].set_axis_off()

    for  i in range(size,len(ax)):
        ax[i].set_visible(False)

    plt.close(fig)
    return fig

def print_auto_encoder(X,net,size,n):

    fig,ax=plt.subplots(nrows=np.math.ceil( size/8 )*2,ncols=8,figsize=(20,20))
    ax=ax.flatten()

    choice=np.random.choice(np.arange(X.shape[0]),size=size)
    fig.tight_layout()
    x=X[choice]

    y_hat=net.predict(np.array(x))

    for pos,i in enumerate(choice):

        ax[pos*2].imshow(X[i].reshape((n,n)),interpolation="nearest",cmap="gray")
        ax[pos*2].set_title(f'original {i}')
        ax[pos*2 +1 ].imshow(y_hat[pos].reshape((n,n)),interpolation="nearest",cmap="gray")
        ax[pos*2 +1 ].set_title(f'reconstuit {i}')
        ax[pos*2].set_axis_off()
        ax[pos*2 +1 ].set_axis_off()

    for  i in range(size*2,len(ax)):
        ax[i].set_visible(False)
        

    plt.close(fig)
    return fig

def load_mist(n_train,n_test):
    d =784
    trs = 60000
    trt = 10000

          
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

 
    X_train = X_train.reshape(trs, d)
    X_test = X_test.reshape(trt, d)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    indx_tr=np.arange(X_train.shape[0])
    indx_tt=np.arange(X_test.shape[0])

    np.random.shuffle(indx_tr)
    np.random.shuffle(indx_tt)

    tr = indx_tr[:n_train]
    tt = indx_tt[:n_test]

    return X_train[tr],y_train[tr],X_test[tt],y_test[tt]

####################################################################################################################################

def add_noise(data,type=0,p=0.1):
    
    if type == 0:
        return data + p * np.random.normal(loc=0.0, scale=0.5, size=data.shape) 
    if type == 1:
        out = data + np.random.choice([0, 1], size=data.shape, p=[1-p, p])
        return np.where(out > 1,1,out)
    else:
        print("wrong type")
