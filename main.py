#!/usr/bin/env python3

import cv2
import time
from display import Display
# import sys
width = 3840//4
height = 2160//4

display = Display(width, height)

def process_frame(image):
    image = cv2.resize(image, (3840//4, 2160//4))
    display.paint(image)
    # cv2.imshow('image', image)
    print(image)
    print(image.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture('videos/scotland.mov')
    if (cap.isOpened() == False):
        print("Error opening video")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
