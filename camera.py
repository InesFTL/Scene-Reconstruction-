import numpy as np
import Conversioncart2hom
import Transformation 

class Camera(object):
    """ Classe de notre camera """

    def __init__(self, P=None, K=None, R=None, t=None):
        """ P = K[R|t] modele camera matrice (3 x 4)
          """
        if P is None:
            try:
                self.extrinsic = np.hstack([R, t])
                P = np.dot(K, self.extrinsic)
            except TypeError as e:
                print('Parametres invalides') #gestion des exceptions
                raise

        self.P = P     # matrice camera
        self.K = K     # matrice intrinseque
        self.R = R     # matrice rotation
        self.t = t     # vecteur translation
        self.c = None  # centre de notre camera
                
                
                #On projete nos point 3D homogène X et on normalise nos coordonnées. On retourne des points projetes en 2D.
    def projection(self, X):
    
        x = np.dot(self.P, X)
        x[0, :] /= x[2, :]
        x[1, :] /= x[2, :]

        return x[:2, :]

    def qr_a_rq_decomposition(self):
      
        Q, R = np.linalg.qr(np.flipud(self.P).T)
        R = np.flipud(R.T)
        return R[:, ::-1], Q.T[::-1, :]

    def factorisation(self):
       
        if self.K is not None and self.R is not None:
            return self.K, self.R, self.t  

        K, R = self.qr_a_rq_decomposition()
       
        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R)  
        self.t = np.dot(np.linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t
    #implementation qui nous retourne le cente de notre camera
    def centre(self):
      
        if self.c is not None:
            return self.c
        elif self.R:
            # implementation de c par factorisation
            self.c = -np.dot(self.R.T, self.t)
        else:
            # P = [M|−MC]
            self.c = np.dot(-np.linalg.inv(self.c[:, :3]), self.c[:, -1])
        return self.c


