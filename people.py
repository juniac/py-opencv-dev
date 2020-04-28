#!/usr/bin/env python3
from collections import deque
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
import cv2
import time
import numpy as np
import argparse
import imutils
from display import Display

width = 3440//4
height = 1440//4
scale = 8
# display = Display(width, height)
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
# rgb(255, 138, 168)
red_lower = (0, 55, 100)
red_upper = (255, 255, 150)
points = deque(maxlen=args["buffer"])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()


def process_frame(image, video_width, video_height):
    frame = cv2.resize(image, (video_width, video_height))
    frame = Rotate(frame, 180)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)

    # Write the output video
    # out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # cv2.imshow("Frame", image)


def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)  # 행렬 변경
        dst = cv2.flip(dst, 1)   # 뒤집기

    elif degrees == 180:
        dst = cv2.flip(src, 0)   # 뒤집기

    elif degrees == 270:
        dst = cv2.transpose(src)  # 행렬 변경
        dst = cv2.flip(dst, 0)   # 뒤집기
    else:
        dst = null
    return dst


if __name__ == "__main__":
    # cap = cv2.VideoCapture('videos/destiny_short.mp4')
    cap = cv2.VideoCapture('videos/people.mov')

    if (cap.isOpened() == False):
        print("Error opening video")
    else:

        video_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / scale))
        video_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / scale))

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:
            process_frame(frame, video_width, video_height)
        else:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

