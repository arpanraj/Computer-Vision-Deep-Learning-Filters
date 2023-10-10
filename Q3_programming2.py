#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 06:35:41 2022

@author: arpanrajpurohit
"""


import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plot


FOLDER_PATH = "/Users/arpanrajpurohit/Desktop/courses/Computer_Vision_and_Deep_Learning_510/Programs/Programming1/"
INPUT_PATH  = FOLDER_PATH + "Q3_input/"
OUTPUT_PATH = FOLDER_PATH + "Q3_output/"
IMAGE_NAMES = ["SIFT1_img.jpg", "SIFT2_img.jpg"]
KEYPOINT_STR = "keypoint_"

image1 = cv.imread(INPUT_PATH + IMAGE_NAMES[0])
image2 = cv.imread(INPUT_PATH + IMAGE_NAMES[1])

gray_image1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)

orb_detector = cv.ORB_create(3000)
keypoints1, descriptors1 = orb_detector.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb_detector.detectAndCompute(image2, None)

img1_with_keypoints = cv.drawKeypoints(image1, keypoints1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_with_keypoints = cv.drawKeypoints(image2, keypoints2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite(OUTPUT_PATH + KEYPOINT_STR + IMAGE_NAMES[0],img1_with_keypoints)
cv.imwrite(OUTPUT_PATH + KEYPOINT_STR + IMAGE_NAMES[1],img2_with_keypoints)

brute_force = cv.BFMatcher()
matched = brute_force.knnMatch(descriptors1, descriptors2, k=2)

best = []

for m,n in matched:
    if m.distance < 0.90*n.distance:
        best.append([m])

gen_image = cv.drawMatchesKnn(image1, keypoints1, image2, keypoints2, best[:100], None, flags= 2)
cv.imwrite(OUTPUT_PATH + "generated.jpg",gen_image)

