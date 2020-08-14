import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from camera import Camera
import MatricesStructures
import Conversioncart2hom
import Appariement
import timeit 

tic=timeit.default_timer() #pour estimer le temps de process de notre code




def reconstruction_3D():
    
    image1 = cv2.imread('imgs/viff.003.ppm')
    image2 = cv2.imread('imgs/viff.001.ppm')
    pts1, pts2 = Appariement.appariements(image1, image2)
    points1 = Conversioncart2hom.cart2hom(pts1)
    points2 = Conversioncart2hom.cart2hom(pts2)

    fig, A_x = plt.subplots(1, 2)
    A_x[0].autoscale_view('tight')
    A_x[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    A_x[0].plot(points1[0], points1[1], 'b.')
    A_x[1].autoscale_view('tight')
    A_x[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    A_x[1].plot(points2[0], points2[1], 'r.')
    fig.show()

    Hauteur, Largeur, ch = image1.shape
    intrinseque = np.array([  
        [2360, 0, Largeur / 2],
        [0, 2360, Hauteur / 2],
        [0, 0, 1]])

    return points1, points2, intrinseque


points1, points2, intrinseque = reconstruction_3D()

# Calcul de la matrice essentielle avec les points plans et normalisation
points1n = np.dot(np.linalg.inv(intrinseque), points1)
points2n = np.dot(np.linalg.inv(intrinseque), points2)
E = MatricesStructures.matrice_essentiel_normalisée(points1n, points2n)
print('Matrice essentielle:', (-E / E[0][1]))
"""
On est au niveau de la caméra 1, on calcule les parametres pour la caméra2
en utilisant la matrice essentielle qui va nous retourner 4 parametres possibles pour la camera
"""

P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2s = MatricesStructures.matrice_P_apartir_essentiel(E)

ind = -1
for i, P2 in enumerate(P2s):
    
    d1 = MatricesStructures.reconstructions_un_point(
        points1n[:, 0], points2n[:, 0], P1, P2)

    #Conversion de P2 du repere camera au repere monde
    P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
    d2 = np.dot(P2_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
tripoints3d = MatricesStructures.lineaire_triangulation(points1n, points2n, P1, P2)
toc=timeit.default_timer()
print('temps ecoule',toc - tic,'secondes') #temps écoulée en secondes

fig = plt.figure()
fig.suptitle('Reconstruction 3D de notre dinosaure ', fontsize=14)
A_x = fig.gca(projection='3d')
A_x.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'r.')
A_x.set_xlabel('X')
A_x.set_ylabel('Y')
A_x.set_zlabel('Z')
A_x.view_init(elev=135, azim=90)
plt.show()
    
