#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 23:20:10 2022

@author: arpanrajpurohit
"""

import random
from math import sqrt, inf
import numpy as np
import matplotlib.pyplot as plot
from PIL import Image


FOLDER_PATH = "/Users/arpanrajpurohit/Desktop/courses/Computer_Vision_and_Deep_Learning_510/Programs/Programming1/"
INPUT_PATH  = FOLDER_PATH + "Q2_input/"
OUTPUT_PATH = FOLDER_PATH + "Q2_output/"
DATASET_PATH = INPUT_PATH + "510_cluster_dataset.txt"
IMAGE_NAMES = ["Kmean_img1.jpg", "Kmean_img2.jpg"]
CLUSTERS = 10
R_ITERATIONS = 2 #random center iterations
A_ITERATIONS = 1 #adjust center iterations
COLORS = ['r', 'g', 'b', 'm', 'c', 'k', 'y','b',
             'g', 'r', 'c']

def create_random_centers(points):
    center_points = []
    indexes = random.sample(range(0,points.shape[0]), CLUSTERS)
    for i in indexes:
        center_points.append(points[i][0])
    return center_points

def find_dis(point1, point2):
    dimentions = len(point1)
    ans = 0
    for dimention in range(dimentions):
        ans += (point1[dimention] - point2[dimention])**2
    return sqrt(ans)

def assign_center(points, center_points):
    for index in range(points.shape[0]):
        min_center_dis = inf
        assigned_center = 0
        for j in range(CLUSTERS):
            current_center_dis = find_dis(points[index][0], center_points[j])
            if min_center_dis > current_center_dis:
                min_center_dis = current_center_dis
                assigned_center = j
        points[index][1] = assigned_center

def adjust_center(points):
    new_centers = []
    for cluster in range(CLUSTERS):
        cluster_point_array = points[points[:,1] == cluster][:,0]
        length_of_cps = cluster_point_array.shape[0]
        center_point_sum = np.sum(cluster_point_array.tolist(), axis=0)
        mean_center_point = center_point_sum/length_of_cps
        new_centers.append(mean_center_point.tolist())
    return new_centers

def find_square_error(points, centers):
    square_mean_error = 0
    for point in points:
        center_index = point[1]
        square_mean_error += find_dis(point[0],centers[center_index])
    return square_mean_error

def convert_to_scatter_points(points):
    x_val = []
    y_val = []
    for point in points:
        x_val.append(point[0])
        y_val.append(point[1])
    return x_val, y_val

def draw_graph(points,centers, square_error, r_iteration, a_iteration):
    for cluster in range(CLUSTERS):
        cluster_point_array = points[points[:,1] == cluster][:,0]
        x, y = convert_to_scatter_points(cluster_point_array)
        plot.scatter(x, y, s=10, c=COLORS[cluster])
    center_x, center_y = convert_to_scatter_points(centers)
    plot.scatter(center_x, center_y, s=300, marker='X', c=COLORS[-1])
    title = " identification - " + str(r_iteration) + "_"+ str(a_iteration) + "Kmeans Graph - SquareError: " + str(round(square_error,2))
    plot.title(title)
    plot.savefig(OUTPUT_PATH + title + ".png")
    plot.show()
        
def find_KMeans_Dataset(points):
    points = np.array(points)
    for r_iteration in range(R_ITERATIONS):
        centers = create_random_centers(points)
        assign_center(points, centers)
        lowest_sum_square_error = [inf, centers]
        for a_iteration in range(A_ITERATIONS):
            centers = adjust_center(points)
            assign_center(points, centers)
            square_error = find_square_error(points,centers)
            draw_graph(points, centers, square_error,r_iteration,a_iteration) 
            if square_error < lowest_sum_square_error[0]:
                lowest_sum_square_error = [square_error, centers]
        print("lowest sum square error is " + str(round(lowest_sum_square_error[0],2)) + " with centers" + str(lowest_sum_square_error[1]))

def generate_image(points, centers, image_shape, image_name):
    c_points = points.copy()
    for i in range(c_points.shape[0]):
        center_index = c_points[i][1]
        c_points[i][0] = centers[center_index]
    image_arr = []
    for j in range(image_shape[0]):
        image_line = []
        for i in range(image_shape[1]):
            image_line.append(c_points[i + j* image_shape[1]][0])
        image_arr.append(image_line)
    filtered_arr = np.array(image_arr).astype(np.uint8)
    filtered_img = Image.fromarray(filtered_arr)
    filtered_img.show()
    filtered_img.save(OUTPUT_PATH + str(CLUSTERS) + image_name)
        
def find_KMeans_Images(points, image_shape, image_name):
    points = np.array(points)
    for r_iteration in range(R_ITERATIONS):
        centers = create_random_centers(points)
        assign_center(points, centers)
        for a_iteration in range(A_ITERATIONS):
            centers = adjust_center(points)
            assign_center(points, centers)
            generate_image(points,centers, image_shape, image_name)
# get data
file = open(DATASET_PATH,'r')
lines = file.readlines()
points = []

for line in lines:
    strings = line.split()
    coordinates = []
    for string in strings:
        coordinates.append(float(string))
    points.append([coordinates, 0])

find_KMeans_Dataset(points)

for image_name in IMAGE_NAMES:
    image = Image.open(INPUT_PATH + image_name)
    image_arr = np.array(image)
    points = []
    for image_line in image_arr:
        for image_pixel in image_line:
            points.append([image_pixel.tolist(),0])
    find_KMeans_Images(points, image_arr.shape, image_name)
    