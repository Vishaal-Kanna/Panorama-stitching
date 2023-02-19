#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
# Add any python libraries here

def patch_generation(img,patch_size):
	patch_range = 0.5
	# Perturbation
	p1 = 32
	p2 = 32
	while('true'):
		A1_x = np.random.randint(int(0.5*patch_range*img.shape[1]), img.shape[1]-int(0.5*patch_range*img.shape[1]))
		A1_y = np.random.randint(int(0.5 * patch_range * img.shape[0]), img.shape[0] - int(0.5 * patch_range * img.shape[0]))
		if A1_x + patch_size<img.shape[1] and A1_y + patch_size<img.shape[0]:
			CA = [(A1_x, A1_y), (A1_x + patch_size, A1_y), (A1_x + patch_size, A1_y + patch_size), (A1_x, A1_y + patch_size)]
			CB = []
			for corner in CA:
				CB.append((corner[0] + np.random.randint(-p1, p1), corner[1] + np.random.randint(-p2, p2)))
			if CB[1][0] < img.shape[1] and CB[1][1] < img.shape[0] and CB[2][0] < img.shape[1] and CB[2][1] < img.shape[0] and CB[3][0] < img.shape[1] and CB[3][1] < img.shape[0]:
				H = cv2.getPerspectiveTransform(np.float32(CA), np.float32(CB))
				if np.linalg.det(H) >0:
					break

	H_inv = np.linalg.inv(H)
	warped_img = cv2.warpPerspective(img, H_inv, (img.shape[1], img.shape[0]))

	patch_A = img[A1_y:A1_y + patch_size, A1_x:A1_x + patch_size]
	patch_B = warped_img[A1_y:A1_y + patch_size, A1_x:A1_x + patch_size]

	X_data = np.dstack((patch_A, patch_B))
	Y_data = np.reshape(np.subtract(np.float32(CA), np.float32(CB)),(8,))
	CA = np.reshape(CA, (8,))
	return X_data, Y_data, patch_A, patch_B, CA

def main():
	img = cv2.imread('/home/vishaal/Vishaal/UMD_Sem_2/CMSC733/vishaal_p1/Phase2/Data/Train/2089.jpg')
	I,L,C,_,_ = patch_generation(img,128)

if __name__ == '__main__':
    main()
 
