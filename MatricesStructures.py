#Construction 

import numpy as np 
import matplotlib.pyplot as plt


def reconstructions_points(p1, p2, m1, m2):
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        res[:, i] = reconstruct_one_point(p1[:, i], p2[:, i], m1, m2)

    return res


def reconstructions_un_point(pt1, pt2, m1, m2):
   
    A = np.vstack([
        np.dot(pre_produit(pt1), m1),
        np.dot(pre_produit(pt2), m2)
    ])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]


def lineaire_triangulation(p1, p2, m1, m2):
    
    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]),
            (p1[1, i] * m1[2, :] - m1[1, :]),
            (p2[0, i] * m2[2, :] - m2[0, :]),
            (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res


def epipole(F):
  
    U, S, V = np.linalg.svd(F)
    e = V[-1]
    return e / e[2]


def epipolaire_lignes(p1, p2, F, show_epipole=False):
    plt.figure()
    plt.suptitle('Lignes epipolaire', fontsize=16)

    plt.subplot(1, 2, 1, aspect='equal')
   
    plot_epipolaire_ligne(p1, p2, F, show_epipole)
    plt.subplot(1, 2, 2, aspect='equal')
    plot_epipolaire_ligne(p2, p1, F.T, show_epipole)


def plot_epipolaire_ligne(p1, p2, F, show_epipole=False):
    """
    On trace l'epipole et la ligne epipolaire a partir de l'équation 
    F*x=0 dans l'image donnee avec les points qui correspondent
    F va etre la matrice fondamentale et p2 le point dans l'autre image
    """
   
    lignes = np.dot(F, p2)
    pad = np.ptp(p1, 1) * 0.01
    mins = np.min(p1, 1)
    maxs = np.max(p1, 1)

    # Parametre de nos lignes epipolaire et leurs valeurs 
    xpts = np.linspace(mins[0] - pad[0], maxs[0] + pad[0], 100)
    for lignes in lignes.T:
        ypts = np.asarray([(ligne[2] + ligne[0] * p) / (-ligne[1]) for p in xpts])
        valid_idx = ((ypts >= mins[1] - pad[1]) & (ypts <= maxs[1] + pad[1]))
        plt.plot(xpts[valid_idx], ypts[valid_idx], linewidth=1)
        plt.plot(p1[0], p1[1], 'ro')

    if vue_epipole:
        epipole = epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def pre_produit(x):
    
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def matrice_P(p2d, p3d):
    
    n = p2d.shape[1]
    if p3d.shape[1] != n:
        raise ValueError('Le nombre de points ne corresponds pas')

    # Creation de la matrice pour la DLT
    M = np.zeros((3 * n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = p3d[:, i]
        M[3 * i + 1, 4:8] = p3d[:, i]
        M[3 * i + 2, 8:12] = p3d[:, i]
        M[3 * i:3 * i + 3, i + 12] = -p2d[:, i]

    U, S, V = np.linalg.svd(M)
    return V[-1, :12].reshape((3, 4))


def matrice_P_apartir_fondamentale(F):
    
    e = epipole(F.T)  
    Te = pre_produit(e)
    return np.vstack((np.dot(Te, F.T).T, e)).T


def matrice_P_apartir_essentiel(E):
    
    U, S, V = np.linalg.svd(E)

    
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # creation de 4 matrices cameras possibles [Hartley page 252]
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s


def correspondances_matrices(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T

    return np.array([
        p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y,
        p1x, p1y, np.ones(len(p1x))
    ]).T


def matrice_image_a_image_matrice(x1, x2, essentiel=False):
    
    A = correspondances_matrices(x1, x2)
  
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if essentiel:
        S = [1, 1, 0] 
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def translation_et_transformation_points(points):
  
    x = points[0]
    y = points[1]
    centre = points.mean(axis=1)  # moyenne de chaque points
    cx = x - centre[0] # centre du points
    cy = y - centre[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * centre[0]],
        [0, scale, -scale * centre[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d


def matrice_normalisée_image_a_image_matrice(p1, p2, essentiel=False):
    
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Le nombre de points ne correspond pas') #gestion des exceptions

    
    p1n, T1 = translation_et_transformation_points(p1)
    p2n, T2 = translation_et_transformation_points(p2)

    
    F = matrice_image_a_image_matrice(p1n, p2n, essentiel)

  
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]


def matrice_fondamentale_normalisé(p1, p2):
    return matrice_normalisée_image_a_image_matrice(p1, p2)


def matrice_essentiel_normalisée(p1, p2):
    return matrice_normalisée_image_a_image_matrice(p1, p2, essentiel=True)
