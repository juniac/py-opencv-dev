#!/usr/bin/env python3

import cv2
import time
# import sys
import sdl2
import sdl2.ext


videoWidth = 3840//4
videoHeight = 2160//4


sdl2.ext.init()


def process_frame(image):
    image = cv2.resize(image, (3840//4, 2160//4))
    events = sdl2.ext.get_events()
    for event in events:
        if event.type == sdl2.SDL_QUIT:
            exit(0)

    # print(dir(window))


    cv2.imshow('image', image)

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
            break;

