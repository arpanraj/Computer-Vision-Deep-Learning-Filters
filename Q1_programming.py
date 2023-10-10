#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 19:35:05 2022

@author: arpanrajpurohit
"""

#load Images
import numpy as np
from PIL import Image
import math

FOLDER_PATH = "/Users/arpanrajpurohit/Desktop/courses/Computer_Vision_and_Deep_Learning_510/Programs/Programming1/"
INPUT_PATH  = FOLDER_PATH + "Q1_unfiltered_images/"
IMAGE_NAMES = ["filter1_img.jpg", "filter2_img.jpg"]

OUTPUT_PATH = FOLDER_PATH + "Q1_filtered_images/"
FILTERED_STR = "_filtered_"
OUTPUT_NAME_EXT = ["3x3" + FILTERED_STR, "5x5"  + FILTERED_STR, "DOGX"  + FILTERED_STR, "DOGY"  + FILTERED_STR, "Sobel"  + FILTERED_STR]
FILTER_PADDING_SIZES = [1, 2, 1, 1, 1]
PAD_MODE = 'constant'
GRAYSCALE_MODE = 'L'
FILTER_COUNT = 5

# 3 x 3 Gaussian

Gauss_3x3_derivative = 1/16
Gauss_3x3_arr        = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
Gauss_3x3_arr_mul    = Gauss_3x3_arr * Gauss_3x3_derivative
Gauss_3x3_arr_mul_shape = Gauss_3x3_arr_mul.shape

def Gaussian_3x3_filter(pade1_image_arr, row, col):
    ans = np.sum(Gauss_3x3_arr_mul * pade1_image_arr[row: row + Gauss_3x3_arr_mul_shape[0], col: col + Gauss_3x3_arr_mul_shape[1]])
    return ans

# 5 x 5 Gaussian

Gauss_5x5_derivative = 1/273
Gauss_5x5_arr        = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
Gauss_5x5_arr_mul    = Gauss_5x5_arr * Gauss_5x5_derivative
Gauss_5x5_arr_mul_shape = Gauss_5x5_arr_mul.shape

def Gaussian_5x5_filter(pade2_image_arr, row, col):
    ans = np.sum(Gauss_5x5_arr_mul * pade2_image_arr[row: row + Gauss_5x5_arr_mul_shape[0], col: col + Gauss_5x5_arr_mul_shape[1]])
    return ans

# Derivative of Gaussian Filter X

DOGx_arr        = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
DOGx_arr_mul    = DOGx_arr 

def DOGx_filter(pade1_image_arr, row, col):
    ans = np.sum(DOGx_arr_mul * pade1_image_arr[row: row + DOGx_arr_mul.shape[0], col: col + DOGx_arr_mul.shape[1]])
    return ans

# Derivative of Gaussian Filter Y

DOGy_arr        = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
DOGy_arr_mul    = DOGy_arr 

def DOGy_filter(pade1_image_arr, row, col):
    ans = np.sum(DOGy_arr_mul * pade1_image_arr[row: row + DOGy_arr_mul.shape[0], col: col + DOGy_arr_mul.shape[1]])
    return ans

# Sobel Filter Y
def Sobel_filter(pade1_image_arr, row, col):
    dogx_value = DOGx_filter(pade1_image_arr, row, col)
    dogy_value = DOGy_filter(pade1_image_arr, row, col)
    dogx_value = dogx_value * dogx_value
    dogy_value = dogy_value * dogy_value
    ans = math.sqrt(dogx_value + dogy_value)
    return ans

for image_name in IMAGE_NAMES:
    image = Image.open(INPUT_PATH + image_name).convert(GRAYSCALE_MODE)
    image_arr = np.array(image)
    
    filters = []
    filter_inputs = []
    for i in range(FILTER_COUNT):
        paded_img_arr = np.pad(image_arr, FILTER_PADDING_SIZES[i], mode=PAD_MODE)
        filter_inputs.append(paded_img_arr)
        filters.append(np.empty(image_arr.shape))
        
    for row in range(image_arr.shape[0]):
        for col in range(image_arr.shape[1]):
            filters[0][row][col] = Gaussian_3x3_filter(filter_inputs[0], row, col)
            filters[1][row][col] = Gaussian_5x5_filter(filter_inputs[1], row, col)
            filters[2][row][col] = DOGx_filter(filter_inputs[2], row, col)
            filters[3][row][col] = DOGy_filter(filter_inputs[3], row, col)
            filters[4][row][col] = Sobel_filter(filter_inputs[4], row, col)
            
    for i in range(FILTER_COUNT):
        filtered_img = Image.fromarray(filters[i]).convert(GRAYSCALE_MODE)
        filtered_img.show()
        filtered_img.save(OUTPUT_PATH + OUTPUT_NAME_EXT[i] + image_name)
        