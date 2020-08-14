import numpy as np
import cv2 #Utilisation d'OpenCV

#Cette fonction nous permet de faire les appariements et de matcher avec la matrice d'homographie



	#On trouve les points appariées et les descripteurs à l'aide de l'algorithme de SIFT
	


	#utilisation de l'algorithme de Flann pour avoir un meilleur appariement
	#trouve sur Opencv


    #Utilisation de la condition de Lowe de l'algorithme de Sift


    # On applique sur l'homographie (xfeatures, FLANN,RANSAC, findHomography sont des modules d'opencv)
   

    # On ne garde que les "bon points" soit les inliers pour de l'algorithme de RANSAC
   

    
def appariements(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create()

    
    ap1, despt1 = sift.detectAndCompute(
        cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
    ap2, despt2 = sift.detectAndCompute(
        cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)

    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(despt1, despt2, k=2)

  
    ok = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            ok.append(m)

    src_pts = np.asarray([ap1[m.queryIdx].pt for m in ok])
    dst_pts = np.asarray([ap2[m.trainIdx].pt for m in ok])

  
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return pts1.T, pts2.T
