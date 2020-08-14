#Permet la conversion de cartesien a homogene et vice-versa 
#Lecture de fichier 

#va lire un fichier contenant une matrice ou chaque ligne represente un point et chaque point est separe d'un espace
#les * sont remplaces par des -1 
import numpy as np 

def read_matrix(path, astype=np.float64):

    with open(path, 'r') as f:
        arr = []
        for line in f:
            arr.append([(token if token != '*' else -1)
                        for token in line.strip().split()])
        return np.asarray(arr).astype(astype)
#chemin : correspond au chemin vers le fichier
#astype : permet de caster les nombres
def cart2hom(arr):
    
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))


def hom2cart(arr):
   
    num_rows = len(arr)
    if num_rows == 1 or arr.ndim == 1:
        return arr

    return np.asarray(arr[:num_rows - 1] / arr[num_rows - 1])
