import pandas as pd
import numpy as np
import cv2 as cv
import sys
import math
import os


path = r"D:\00_Study\IS\02_code\00_dataset\05_test_red_hue\original"
fileLists = os.listdir(path)

tileLists = []

for i in range(fileLists):

    # Reading image
    img =  cv.imread(r"D:\00_Study\IS\02_code\00_dataset\05_test_red_hue\original\{}.jpg".format(i))

    height=img.shape[0]
    width=img.shape[1]

    y1 = 0
    M = math.ceil(height/5)
    N = math.ceil(width/5)

    for y in range(0,height,M):
        for x in range(0, width, N):
            y1 = y + M
            x1 = x + N
            tiles = img[y:y+M,x:x+N]
            tileLists.append(tiles)
