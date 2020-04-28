#!/usr/bin/env python3
from collections import deque
from imutils.video import VideoStream
import cv2
import time
import numpy as np
import argparse
import imutils
from display import Display

width = 3440//4
height = 1440//4
scale = 4
# display = Display(width, height)
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
# rgb(255, 138, 168)
red_lower = (5, 190, 5)
red_upper = (255, 250, 255)
points = deque(maxlen=args["buffer"])


def process_frame(image, video_width, video_height):

    image = cv2.resize(image, (video_width, video_height))
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, red_lower, red_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # cv2.imshow("Frame", mask)

    counts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    counts = imutils.grab_contours(counts)
    center = None

    if len(counts) > 0:
        c = max(counts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)

    points.appendleft(center)

    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(image, points[i - 1], points[1], (0, 0, 255), thickness)

    cv2.imshow("Frame", image)


if __name__ == "__main__":
    # cap = cv2.VideoCapture('videos/destiny_short.mp4')
    cap = cv2.VideoCapture('videos/red_something.MOV')

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


