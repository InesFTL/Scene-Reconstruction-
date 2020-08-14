#Cette fonction nous permet de créer différentes matrices de rotations
import numpy as np

def matrice_rotation(x_angle, y_angle=0, z_angle=0):
	A_x = np.deg2rad(x_angle)
	A_y = np.deg2rad(y_angle)
	A_z = np.deg2rad(z_angle)

	#Rotation autour de l'axe x

	R_x = np.array([
		 [1, 0, 0],
        [0, np.cos(A_x), -np.sin(A_x)],
        [0, np.sin(A_x), np.cos(A_x)]
		])

	#Rotation  autour de l'axe y

	R_y = np.array([
		[np.cos(A_y), 0, np.sin(A_y)],
        [0, 1, 0],
        [-np.sin(A_y), 0, np.cos(A_y)]
		])

	#Rotation autour de l'axe z 
	R_z = np.array([
		[np.cos(A_z), -np.sin(A_z), 0],
        [np.sin(A_z), np.cos(A_z), 0],
        [0, 0, 1]
		])

	return np.dot(np.dot(R_x,R_y),R_z)