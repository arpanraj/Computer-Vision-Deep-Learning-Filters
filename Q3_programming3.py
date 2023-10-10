#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 21:44:11 2022

@author: arpanrajpurohit
"""


import torch
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plot

from copy import deepcopy


FOLDER_PATH = "/Users/arpanrajpurohit/Desktop/courses/Computer_Vision_and_Deep_Learning_510/Programs/Programming1/"
INPUT_PATH  = FOLDER_PATH + "Q3_input/"
OUTPUT_PATH = FOLDER_PATH + "Q3_output/"
IMAGE_NAMES = ["SIFT1_img.jpg", "SIFT2_img.jpg"]
KEYPOINT_STR = "keypoint_"

def euc_distance_mat(desc1, desc2):
    dis1_sq = torch.sum(desc1 * desc1, dim=1).unsqueeze(-1)
    dis2_sq = torch.sum(desc2 * desc2, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((dis1_sq.repeat(1, desc2.size(0)) + torch.t(dis2_sq.repeat(1, desc1.size(0)))
                      - 2.0 * torch.bmm(desc1.unsqueeze(0), torch.t(desc2).unsqueeze(0)).squeeze(0))+eps)

image1 = cv.imread(INPUT_PATH + IMAGE_NAMES[0])
image2 = cv.imread(INPUT_PATH + IMAGE_NAMES[1])

gray_image1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
keypoints1, des1 = sift.detectAndCompute(gray_image1,None)
keypoints2, des2 = sift.detectAndCompute(gray_image2,None)

img_with_keypoints1 =cv.drawKeypoints(gray_image1,keypoints1,image1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_with_keypoints2 =cv.drawKeypoints(gray_image2,keypoints2,image2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite(OUTPUT_PATH + KEYPOINT_STR + IMAGE_NAMES[0],img_with_keypoints1)
cv.imwrite(OUTPUT_PATH + KEYPOINT_STR + IMAGE_NAMES[1],img_with_keypoints2)

device = torch.device('cpu')

def draw_final_image(keypoints1, keypoints2, tentatives, image1, image2):
    best = []
    for i in range(len(tentatives)):
        best.append(cv.DMatch(tentatives[i,0],tentatives[i,1], 1))
    fst_img_pts = np.float32([ keypoints1[m[0]].pt for m in tentatives ]).reshape(-1,1,2)
    snd_img_pts = np.float32([ keypoints2[m[1]].pt for m in tentatives ]).reshape(-1,1,2)
    Ha, mask = cv.findHomography(fst_img_pts, snd_img_pts, cv.RANSAC,1.0)
    if Ha is None:
        print ("No homography found")
        return
    matchesMask = mask.ravel().tolist()
    h,w,ch = image1.shape
    points = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dest = cv.perspectiveTransform(points, Ha)
    img2_tr = cv.polylines(gray_image2,[np.int32(dest)],True,(0,0,255),3, cv.LINE_AA)
    
    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    outimg = cv.drawMatches(gray_image1,keypoints1,img2_tr,keypoints2,best,None,**draw_params)
    cv.imwrite(OUTPUT_PATH + "generated2.jpg",outimg)
    plot.imshow(outimg)
    return

def nearest_neighbour(desc1, desc2):
    distances = euc_distance_mat(torch.from_numpy(desc1.astype(np.float32)).to(device),
                        torch.from_numpy(desc2.astype(np.float32)).to(device))
    values, ids_in_2 = torch.topk(distances, 2 ,dim=1, largest=False)
    cast = (values[:,0] / values[:,1]) <= 0.9
    ids_in1 = torch.arange(0, ids_in_2.size(0))[cast]
    ids_in_2 = ids_in_2[:,0][cast]
    matches_ids = torch.cat([ids_in1.view(-1,1), ids_in_2.cpu().view(-1,1)],dim=1)
    return matches_ids.cpu().data.numpy()

nearest_neighbour = nearest_neighbour(des1, des2)
draw_final_image(keypoints1, keypoints2, nearest_neighbour, image1, image2)
