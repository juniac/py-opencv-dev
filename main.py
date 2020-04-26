#!/usr/bin/env python3

import cv2
import time
import numpy as np
from display import Display
# import sys
width = 3840//4
height = 2160//4

display = Display(width, height)
orb = cv2.ORB_create()

class FeatureExtractor(object):
    GX = 16 // 2
    GY = 12 // 2
    
    def __init__(self, width, height):
        self.orb = cv2.ORB_create(1000)
        self.width, self.height = width, height

    def extract(self, image):
       features = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
       print(features)
       return features
       """  
        sy = image.shape[0] // self.GY
        sx = image.shape[1] // self.GX
        # print([image.shape[0], image.shape[1]])
        a_key_points = []
        for ry in range(0, image.shape[0], sy):
            for rx in range(0, image.shape[1], sx):
                image_chunk = image[ry:ry + sy, rx:rx + sx]
                # print(image_chunk)
                key_points = self.orb.detect(image_chunk, None)
                print(key_points)
                for point in key_points:
                    point.pt = (point.pt[0] + rx, point.pt[1] + ry)
                    a_key_points.append(point)
                    # print(point.pt)
                    # print(point)
        return a_key_points 
        """
                

feature_extractor = FeatureExtractor(width, height)

def process_frame(image):
    image = cv2.resize(image, (3840//4, 2160//4))
    key_points = feature_extractor.extract(image)

    for f in key_points:
        x, y = map(lambda x: int(round(x)), f[0])
        cv2.circle(image, (x,y), color=(0, 255, 0), radius=3)
    # for point in key_points:
    #     x, y = map(lambda x: int(round(x)), point.pt)
    #     cv2.circle(image, (x,y), color=(0, 255, 0), radius=3)
        # print([x, y])
    display.paint(image)
    # cv2.imshow('image', image)
    # print(image)
    # print(image.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture('videos/kyushu.mov')
    if (cap.isOpened() == False):
        print("Error opening video")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
