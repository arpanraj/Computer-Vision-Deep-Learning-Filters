#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 23:20:10 2022

@author: arpanrajpurohit
"""

import numpy as np
from PIL import Image
import cv2 as cv


FOLDER_PATH = "/Users/arpanrajpurohit/Desktop/courses/Computer_Vision_and_Deep_Learning_510/Programs/Programming1/"
INPUT_PATH  = FOLDER_PATH + "Q3_input/"
OUTPUT_PATH = FOLDER_PATH + "Q3_output/"
IMAGE_NAMES = ["SIFT1_img.jpg", "SIFT2_img.jpg"]
KEYPOINT_STR = "keypoint_"

image1 = cv.imread(INPUT_PATH + IMAGE_NAMES[0])
image2 = cv.imread(INPUT_PATH + IMAGE_NAMES[1])

gray_image1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
keypoints1 = sift.detect(gray_image1,None)
keypoints2 = sift.detect(gray_image2,None)

gen_image1 = cv.drawKeypoints(gray_image1,keypoints1,image1)
gen_image2 = cv.drawKeypoints(gray_image2,keypoints2,image2)

cv.imwrite(OUTPUT_PATH + KEYPOINT_STR + IMAGE_NAMES[0],gen_image1)
cv.imwrite(OUTPUT_PATH + KEYPOINT_STR + IMAGE_NAMES[1],gen_image2)