import pandas as pd
import numpy as np
import cv2 as cv
from itertools import chain
import math
import os

def masking(img, color):

    if color == 'red':

        bound1 = cv.inRange(img, np.array([160,0,0]), np.array([179,255,255]))
        bound2 = cv.inRange(img, np.array([0,0,0]), np.array([15,255,255]))

        mask = bound1 + bound2
    
    elif color == 'orange':

        bound1 = cv.inRange(img, np.array([170,0,0]), np.array([179,255,255]))
        bound2 = cv.inRange(img, np.array([0,0,0]), np.array([25,255,255]))

        mask = bound1 + bound2

    elif color == 'yellow':
        mask = cv.inRange(img, np.array([20,0,0]), np.array([40,255,255]))

    elif color == 'green':
        mask = cv.inRange(img, np.array([30,0,0]), np.array([80,255,255]))

    elif color == 'aqua':
        mask = cv.inRange(img, np.array([75,0,0]), np.array([95,255,255]))

    elif color == 'blue':
        mask = cv.inRange(img, np.array([90,0,0]), np.array([145,255,255]))

    elif color == 'purple':
        mask = cv.inRange(img, np.array([100,0,0]), np.array([160,255,255]))

    else:
        mask = cv.inRange(img, np.array([150,0,0]), np.array([170,255,255]))

    return mask

def makeFrequencyDist(hist):

    if np.sum(np.sum(hist)) == 0:
        frequency = np.array([0])
    else:
        frequency = np.repeat(np.arange(hist.shape[0]), hist.ravel().astype(int))

    return frequency

def calMean(arr):
    mean = np.mean(arr)
    return mean

def calMedian(arr):
    median = np.median(arr)
    return median

def calStd(arr):
    std = np.std(arr)
    return std

def calMax(arr):
    max = np.max(arr)
    return max

def setColName(df):

    a = ['mean', 'median', 'std', 'max']
    properties = (a*3 + ['area']) * 16

    a = ['hue', 'lum', 'sat']
    b = [4]*3
    channel = list(chain.from_iterable([x]*y for x,y in zip(a,b)))
    channel = (channel + ['area']) * 16

    a = ['red', 'orange', 'yellow', 'green', 
            'aqua', 'blue', 'purple', 'magenta']
    b = [13]*8
    color = list(chain.from_iterable([x]*y for x,y in zip(a,b)))
    color = color*2

    a = ['before', 'after']
    b = [104]*2
    imgPair = list(chain.from_iterable([x]*y for x,y in zip(a,b)))

    columns = pd.MultiIndex.from_arrays([imgPair, color, channel, properties])
    df.columns = columns
    return df


path = r"D:\00_Study\IS\02_code\00_dataset\07_test_red_hue_3\test_input"
fileLists = os.listdir(path)

#### for test dataset ####
fileLists = sorted(list(map(lambda x: int(x[:-4]), fileLists)))

tileListsBefore = []
tileListsAfter = []
df = []

colorList = ['red', 'orange', 'yellow', 'green',
            'aqua', 'blue', 'purple', 'magenta']

# for i in range(len(fileLists)):
# for i in range(100):
for i in fileLists:
    print(i)
    # Reading image
#     imgBefore =  cv.imread(r"D:\00_Study\IS\02_code\00_dataset\07_test_red_hue_3\test_input\{}.jpg".format(i))
#     imgAfter =  cv.imread(r"D:\00_Study\IS\02_code\00_dataset\07_test_red_hue_3\test_actual\{}.jpg".format(i))

#     heightBefore = imgBefore.shape[0]
#     widthBefore = imgBefore.shape[1]
#     heightAfter = imgAfter.shape[0]
#     widthAfter = imgAfter.shape[1]

#     y1 = 0
#     x1 = 0
#     M = math.ceil(heightBefore/5)
#     N = math.ceil(widthBefore/5)

#     for y in range(0,heightBefore,M):
#         for x in range(0, widthBefore, N):
#             y1 = y + M
#             x1 = x + N
#             tileBefore = imgBefore[y:y+M,x:x+N]
#             tileListsBefore.append(tileBefore)
#     x1 = 0
#     y1 = 0
#     M = math.ceil(heightAfter/5)
#     N = math.ceil(widthAfter/5)
    
#     for y in range(0,heightAfter,M):
#         for x in range(0, widthAfter, N):
#             y1 = y + M
#             x1 = x + N
#             tileAfter = imgAfter[y:y+M,x:x+N]
#             tileListsAfter.append(tileAfter)


# for i in range(len(tileListsBefore)):
#     print(i)
#     # select image path
#     imgBefore = tileListsBefore[i]
#     imgAfter = tileListsAfter[i]
#     imgSize = imgBefore.shape[0] * imgBefore.shape[1]

#     # Convert to HLS
#     imgBefore = cv.cvtColor(imgBefore, cv.COLOR_BGR2HLS)
#     imgAfter = cv.cvtColor(imgAfter, cv.COLOR_BGR2HLS)

#     imgPair = [imgBefore, imgAfter]
#     imgProperties = []

#     # get histogram for each channel
#     for img in imgPair:

#         # masking with each color
#         for color in colorList:

#             mask = masking(imgBefore, color)
#             maskArea = masking(img, color)
#             maskArea = maskArea[maskArea != 0].shape[0] / imgSize
#             colorProperties = []
        
#             hue = cv.calcHist([img], [0], mask, [180], [0,180])
#             lum = cv.calcHist([img], [1], mask, [256], [0,256])
#             sat = cv.calcHist([img], [2], mask, [256], [0,256])

#             imgChannels = [hue, lum, sat]

#             for channel in imgChannels:
                
#                 frequency = makeFrequencyDist(channel)

#                 channelPropeties = [
#                     calMean(frequency),
#                     calMedian(frequency),
#                     calStd(frequency),
#                     calMax(frequency)]
                
#                 colorProperties.append(channelPropeties)

#             colorProperties = list(chain.from_iterable(colorProperties))
#             colorProperties.append(maskArea)
#             imgProperties.append(colorProperties)

#     df.append(list(chain.from_iterable(imgProperties)))

# df = pd.DataFrame(df)
# setColName(df).to_csv('210922_1K_img_split_25_blocks.csv')