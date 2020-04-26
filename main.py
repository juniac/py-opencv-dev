#!/usr/bin/env python3

import cv2
import time
import numpy as np
from display import Display
from extractor import FeatureExtractor
# import sys
width = 3840//4
height = 2160//4

display = Display(width, height)
feature_extractor = FeatureExtractor()
# orb = cv2.ORB_create()

def process_frame(image):
    image = cv2.resize(image, (3840//4, 2160//4))
    matches = feature_extractor.extract(image)

    for point1, point2 in matches:
        x, y = map(lambda x: int(round(x)), point1.pt)
        match_x, match_y = map(lambda x: int(round(x)), point2.pt)
        cv2.circle(image, (x,y), color=(0, 255, 0), radius=3)
        cv2.line(image, (x, y), (match_x, match_y), color=(255, 0, 0))
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
