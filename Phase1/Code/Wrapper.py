#!/usr/bin/evn python

"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Code

Author(s): 
Pavan Mantripragada (mppavan@umd.edu) 
Masters in Robotics,
University of Maryland, College Park

Vishaal Kanna (vishaal@umd.edu) 
Masters in Robotics,
University of Maryland, College Park
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from random import sample
from skimage.feature import peak_local_max
import argparse

class Image:
	def __init__(self,image):
		self.bgr = image.copy()
		self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		self.harris_cmap = None
		self.corners = None
		self.features = None
		self.feature_imgs = None
		self.feature_descriptors = None
	
	def get_harris_cmap(self,savepath,sup_threshold=0.01):
		gray_img = np.float32(self.gray)
		C_img = cv2.cornerHarris(gray_img,2,3,0.001)
		C_img[C_img<sup_threshold*C_img.max()] = 0
		self.harris_cmap = C_img.copy()
		harris_corner_img = copy.copy(self.bgr)
		harris_corner_img[C_img > 0]=[0,0,255]
		cv2.imwrite(savepath,harris_corner_img)
		cv2.namedWindow("Harris Corners", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Harris Corners", 1280, 720)
		cv2.imshow('Harris Corners',harris_corner_img)
		cv2.waitKey(500)
		cv2.destroyAllWindows()

	def cornerANMS(self,savepath,N_best=1000,min_distance=9):
		coords = peak_local_max(self.harris_cmap,min_distance)
		N_strong = coords.shape[0]
		x = coords[:,1].reshape(-1,)
		y = coords[:,0].reshape(-1,)
		r = [float("inf") for _ in range(N_strong)]
		ED = 0
		for i in range(N_strong):
			for j in range(N_strong):
				if self.harris_cmap[y[j],x[j]] > self.harris_cmap[y[i],x[i]]:
					ED = (x[j]-x[i])**2 + (y[j]-y[i])**2
				if ED < r[i]:
					r[i] = ED
		if N_strong < N_best:
			N_best = N_strong
			print("ANMS N_best reduced to ", N_best)

		idx = np.flip(np.argsort(r))[0:N_best]
		x_best = x[idx].astype(int)
		y_best = y[idx].astype(int)
		anms_corner_img = self.bgr.copy()
		for i in range(x_best.shape[0]):
			cv2.circle(anms_corner_img, (x_best[i], y_best[i]), 5, (0, 255, 0), -1)
		self.corners = np.hstack((x_best.reshape(-1,1),y_best.reshape(-1,1))).tolist()
		cv2.imwrite(savepath,anms_corner_img)
		cv2.namedWindow("Suppressed Corners", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Suppressed Corners", 1280, 720)	
		cv2.imshow('Suppressed Corners',anms_corner_img)
		cv2.waitKey(500)
		cv2.destroyAllWindows()	
		
	def get_features(self,savepath,l=40):
		r = 4/l
		self.features = []
		self.feature_imgs = []
		self.feature_descriptors = []
		for x,y in self.corners:
			h,w = self.gray.shape
			if x-l>=0 and x+l<=w and y-l>=0 and y+l<=h:  
				patch = self.gray[y-l:y+l,x-l:x+l]
				self.features.append([x,y])
				patch = cv2.GaussianBlur(patch,(3,3),1)
				patch = cv2.resize(patch, None, fx=r, fy=r, interpolation = cv2.INTER_CUBIC)
				patch = (patch-patch.mean())/patch.std()
				self.feature_imgs.append(patch)
				feature_descriptor = patch.reshape(-1,)
				self.feature_descriptors.append(feature_descriptor)
		pretty_print_FD(self.feature_imgs,savepath,10,10)		

	def match(self,other,alpha=0.8):
		matches = []
		for idx1,FD1 in enumerate(self.feature_descriptors):
			ssds = []
			for idx2,FD2 in enumerate(other.feature_descriptors):
				diff = FD1-FD2
				dist = np.dot(diff,diff)
				ssds.append(dist)
			idxs = np.argsort(ssds)
			idx2_min = idxs[0]
			idx2_2min = idxs[1]
			low_dist = ssds[idx2_min]
			sec_low_dist = ssds[idx2_2min]
			if low_dist/sec_low_dist < alpha:
				matches.append([self.features[idx1][0],self.features[idx1][1],
								other.features[idx2_min][0],other.features[idx2_min][1]])
		H, better_matches = ransac(matches)
		return better_matches, matches, H



class ImageSet:
	def __init__(self,image_list,savepath):
		self.num_imgs = len(image_list)
		self.savepath = savepath
		self.checklist = [{} for _ in range(self.num_imgs)]
		self.island_map = [0 for _ in range(self.num_imgs)]
		self.imgs = []
		for image in image_list:
			self.imgs.append(Image(image))
		self.connected_imgs = []

	def convert(self,from_island,to_island):
		for i in range(self.num_imgs):
			if self.island_map[i] == from_island:
				self.island_map[i] = to_island

	def connect(self):
		for i in range(self.num_imgs):
			if self.island_map[i] == 0:
				self.island_map[i] = max(self.island_map)+1
			current_island = self.island_map[i]
			other_ids = [i for i in range(self.num_imgs) if self.island_map[i] != current_island]
			for j in other_ids:
				if j not in self.checklist[i]:
					print("matching images : ",i+1,j+1)
					matches, all_matches, H = self.imgs[i].match(self.imgs[j])
					if quality(matches):
						savepath = self.savepath + str(i+1) + "_" + str(j+1) + "_matches.png"
						drawmatches(self.imgs[i].bgr,self.imgs[j].bgr,all_matches,savepath)
						savepath = self.savepath + str(i+1) + "_" + str(j+1) + "_better_matches.png" 
						drawmatches(self.imgs[i].bgr,self.imgs[j].bgr,matches,savepath)						
						debugging_flag = True
						self.checklist[i][j] = (True,H,matches,all_matches)
						H,matches,all_matches = flip(H,matches,all_matches,debugging_flag)
						self.checklist[j][i] = (True,H,matches,all_matches)
						other_island = self.island_map[j]
						if other_island == 0:
							self.island_map[j] = current_island
						else:
							from_island = max(current_island,other_island)
							to_island = min(current_island,other_island)
							self.convert(from_island,to_island)
						break
					else:
						debugging_flag = False
						self.checklist[i][j] = (False,H,matches,all_matches)
						H,matches,all_matches = flip(H,matches,all_matches,debugging_flag)
						self.checklist[j][i] = (False,H,matches,all_matches)
		for i in range(self.num_imgs):
			if self.island_map[i] == 1:
				self.connected_imgs.append(i)

	def max_depth(self,prev,visited,curr_d):
		depths = []
		for id in self.checklist[prev].keys():
			if self.checklist[prev][id][0]:
				if id not in visited:
					new_visited = visited.copy()
					new_visited[id] = True
					new_d = 1 + curr_d 
					depths.append(self.max_depth(id,new_visited,new_d))
		if len(depths) == 0:
			return curr_d
		return max(depths)

	def traverse(self,prev,visited,H):
		H_dict = {}
		for id in self.checklist[prev].keys():
			if self.checklist[prev][id][0]:
				if id not in visited:
					new_visited = visited.copy()
					new_visited[id] = True
					H_dict[id] = H @ self.checklist[id][prev][1]
					H_dict.update(self.traverse(id,new_visited,H_dict[id]))
		return H_dict

	def get_center(self):
		e = []
		for i in self.connected_imgs:
			vis = {i:True}
			e.append(self.max_depth(i,vis,0))
		return e.index(min(e))

	def get_stiching_inputs(self):
		base_id = self.get_center()
		H_dict = self.traverse(base_id,{base_id:True},np.identity(3))
		imgb = self.imgs[base_id].bgr
		imgs = []
		Hs = []
		for id,H in H_dict.items():
			imgs.append(self.imgs[id].bgr)
			Hs.append(H)
		return imgb,imgs,Hs

def flip(H,matches,all_matches,debugging_flag):
	try:
		H = np.linalg.inv(H)
		H /= H[2,2]
	except np.linalg.LinAlgError:
		if debugging_flag:
			print("Concern!!!!!!!!!!!")
		H = np.identity(3)
	matches = np.array(matches,dtype=int)
	temp = matches[:,0:2].copy()
	matches[:,0:2] = matches[:,2:4].copy()
	matches[:,2:4] = temp.copy()
	all_matches = np.array(all_matches,dtype=int)
	temp = all_matches[:,0:2].copy()
	all_matches[:,0:2] = all_matches[:,2:4].copy()
	all_matches[:,2:4] = temp.copy()
	matches = matches.tolist()
	all_matches = all_matches.tolist()
	return H,matches,all_matches

def quality(matches):
	if len(matches) < 10:
		return False
	else:
		return True

def sort_names(names):
	ids = []
	for name in names:
		id = int(name.split(".")[0])
		ids.append(id)
	ids = np.array(ids,dtype=int)
	i = np.argsort(ids)
	nom = np.array(names)
	sorted_names = nom[i]
	return sorted_names.tolist()

def pretty_print_FD(FD_imgs,savepath,rows,cols):
	fig,ax = plt.subplots(rows, cols, figsize=(cols,rows))
	ax = np.ravel(ax)
	if len(FD_imgs) > 100:
		n = 100
	else:
		n = -1
	for idx,filter in enumerate(FD_imgs[0:n]):
		ax[idx].imshow(filter, cmap='gray')
		ax[idx].axis('off')
	plt.tight_layout()
	plt.savefig(savepath,format="png")
	plt.close(fig)	

def get_eight_random_points(img1_points,img2_points):
	n = img1_points.shape[0]
	if n <= 4:
		return True, [], []
	nums = sample(range(0, n), 4)
	p1 = img1_points[nums]
	p2 = img2_points[nums]
	return False, p1.astype(np.float32), p2.astype(np.float32)

def check_singular(points):
	points = np.array(points)
	n = points.shape[0]
	if n < 2:
		return True
	for i in range(40):
		nums = sample(range(0, n), 2)
		p = points[nums]
		if np.linalg.norm(p[0]-p[1]) < 1e-7:
			return True
	return False

def ransac(matches):
	img1_points = np.array(matches,dtype=np.float32)[:,0:2] 
	img2_points = np.array(matches,dtype=np.float32)[:,2:4]
	n = 0
	Hs = None
	eps = 10
	iter = 5000
	ones = np.ones((img1_points.shape[0],1),dtype=np.float32)
	X1 = np.hstack((img1_points,ones))
	X2 = np.hstack((img2_points,ones))
	points1_idx = np.array([[0,0]],dtype=int)
	points2_idx = np.array([[0,0]],dtype=int)
	for i in range(0,iter):
		flag, p1, p2 = get_eight_random_points(img1_points, img2_points)
		if flag:
			continue
		H = cv2.getPerspectiveTransform(p1,p2)
		
		point_set1 = []
		point_set2 = []
		for j in range(0,img1_points.shape[0]):
			Xp = H @ X1[j].T
			Xp /= Xp[2]
			diff = X2[j] - Xp 
			SSD = np.dot(diff,diff)
			if SSD < eps:
				point_set1.append(img1_points[j])
				point_set2.append(img2_points[j])
		if check_singular(point_set2):
			continue
		if n < len(point_set1):
			n = len(point_set1)
			Hs = H
			points1_idx = np.array(point_set1,dtype=int)
			points2_idx = np.array(point_set2,dtype=int)
	print(f"{n} inliers found in feature points")
	better_matches = np.hstack((points1_idx,points2_idx)).tolist()
	return Hs, better_matches
	

def drawmatches(img1,img2,matches,savepath):
	num_matches = len(matches)
	points = np.array(matches,dtype=float)
	kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in points[:,0:2]]
	kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in points[:,2:4]]
	good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx, _distance=0) \
	 				for idx in range(num_matches)]
	out_img = np.array([])
	out_img = cv2.drawMatches(img1, kp1, img2, kp2,
							  good_matches, out_img)
	
	cv2.imwrite(savepath, out_img)
	cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Matches", 1280, 720) 
	cv2.imshow("Matches", out_img)
	cv2.waitKey(500) 
	cv2.destroyAllWindows()

def get_range(wb,hb,w,h,Hs):
	ptsb = np.float32([[0,0],[0,hb],[wb,hb],[wb,0]]).reshape(-1,1,2)
	pts_ = []
	for i in range(len(h)):
		ptsi = np.float32([[0,0],[0,h[i]],[w[i],h[i]],[w[i],0]]).reshape(-1,1,2)
		ptsi_ = cv2.perspectiveTransform(ptsi, Hs[i])
		pts_.append(ptsi_)
	
	pts = ptsb.copy()
	for ptsi_ in pts_:
		pts = np.concatenate((pts, ptsi_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
	t = [-xmin,-ymin]
	w_res = xmax-xmin
	h_res = ymax-ymin
	return t,w_res,h_res
	
def autopano(imgb, imgs, Hs):
	"""
	imgb - center image
	imgs - list of all other images
	Hs - list of homographies to warp imgs to imgb
	"""
	print("Stitching image")
	while True:
		hb,wb = imgb.shape[:2]
		h = []
		w = []
		for i,img in enumerate(imgs):
			hi,wi = img.shape[:2]
			h.append(hi)
			w.append(wi)
		t,w_res,h_res = get_range(wb,hb,w,h,Hs)
		if w_res > 9000 or h_res > 7000:
			# Rescaling
			s1 = 0.1
			s2 = 0.1
			S = np.diag([s1,s2,1])
			Si = np.diag([1/s1,1/s2,1])
			imgb = cv2.resize(imgb, None, fx=s1, fy=s2, interpolation = cv2.INTER_CUBIC)
			for i in range(len(imgs)):
				imgs[i] = cv2.resize(imgs[i], None, fx=s1, fy=s2, interpolation = cv2.INTER_CUBIC)
				Hs[i] = S @ Hs[i] @ Si
		else:
			break

	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
	pano = np.zeros((h_res, w_res, 3),dtype=np.uint8)
	pano[t[1]:hb+t[1],t[0]:wb+t[0]] = imgb
	for i in range(len(h)):
		print("about to warp img ",i+1)
		result = cv2.warpPerspective(imgs[i], Ht.dot(Hs[i]), (w_res, h_res))
		pano = np.maximum(pano,result)
	
	return pano

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--datatype', default="Train", help='Input Train or Test')
	Parser.add_argument('--dataset', default="Set1", help='Input dataset name ex: Set1')
	Args = Parser.parse_args()

	"""
	Read a set of images for Panorama stitching
	"""
	set_name = Args.dataset +  "/"
	data_sub_folder = Args.datatype + "/"
	data_path = "../Data/" + data_sub_folder + set_name
	save_path = "../Results/" + data_sub_folder + set_name
	image_name_list = os.listdir(data_path)
	image_name_list = sort_names(image_name_list)

	bgr_img_list = []
	for img_name in image_name_list:
		print("Loading Image : " + img_name)
		bgr_img_list.append(cv2.imread(data_path + img_name))

	pano_set = ImageSet(bgr_img_list,save_path)
	
	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	for idx,img in enumerate(pano_set.imgs):
		filename = str(idx+1) + "_corners.png"
		corner_img_savename = save_path + filename
		img.get_harris_cmap(corner_img_savename)

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
	for idx,img in enumerate(pano_set.imgs):
		filename = str(idx+1) + "_anms.png"
		corner_img_savename = save_path + filename
		img.cornerANMS(corner_img_savename)

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
	print("Extracting Features...")
	for idx,img in enumerate(pano_set.imgs):
		filename = str(idx+1) + "_FD.png"
		FD_savename = save_path + filename
		img.get_features(FD_savename)
	print("Feature descriptors are saved in results folder please check there!")
	
	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""
	"""
	Refine: RANSAC, Estimate Homography
	"""
	pano_set.connect()
	print("connected images")
	imgb,imgs,Hs = pano_set.get_stiching_inputs()
	print("got stiching inputs")

	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	pano = autopano(imgb, imgs, Hs)
	filename = "mypano.png"
	mypano_savename = save_path + filename
	cv2.imwrite(mypano_savename,pano)
	cv2.namedWindow("result", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("result", 1280, 720)		
	cv2.imshow("result", pano)
	cv2.waitKey() 
	cv2.destroyAllWindows()

		

    
if __name__ == '__main__':
    main()
 
