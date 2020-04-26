#!/usr/bin/env python3

import cv2
import time
from display import Display
# import sys
width = 3840//4
height = 2160//4

display = Display(width, height)
orb = cv2.ORB_create()


def process_frame(image):
    image = cv2.resize(image, (3840//4, 2160//4))

    keyPoints, destination = orb.detectAndCompute(image, None)

    for point in keyPoints:
        # print(point.pt)
        x, y = map(lambda x: int(round(x)), point.pt)
        cv2.circle(image, (x,y), color=(0, 255, 0), radius=3)
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
