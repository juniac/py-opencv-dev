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
scale = 2
# display = Display(width, height)
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
# rgb(255, 138, 168)
red_lower = np.array([151, 145, 44])
red_upper = np.array([249, 255, 255])
points = deque(maxlen=args["buffer"])


def process_frame(image, video_width, video_height):


    start_x, start_y = 150, 30
    crop_width, crop_height = video_width - 300, video_height - 100
    image = cv2.resize(image, (video_width, video_height))
    image = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, red_lower, red_upper)

    # red = cv2.bitwise_and(image, image, mask=mask)



    # mask = cv2.erode(mask, None, iterations=2)
    # cv2.imshow("Frame", mask)
    # mask = cv2.dilate(mask, None, iterations=2)


    # sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharpen = cv2.filter2D(blurred, -1, sharpen_kernel)

    # thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # cv2.imshow("Frame", thresh)

    counts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    counts = imutils.grab_contours(counts)
    center = None

    if len(counts) > 0:
        c = max(counts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > 20:
            cv2.rectangle(image, (x, y), (x + w, y + h + 100), (36, 255, 12), 2)
        # cv2.circle(image, (int(x), int(y)), int(10), (0, 255, 255), 2)


    cv2.imshow("Frame", image)
        # M = cv2.moments(c)
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # if radius > 10:
            # cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # cv2.circle(image, center, 5, (0, 0, 255), -1)

    # points.appendleft(center)

    # for i in range(1, len(points)):
    #     if points[i - 1] is None or points[i] is None:
    #         continue

    #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    #     cv2.line(image, points[i - 1], points[1], (0, 0, 255), thickness)

''' 
 '''

if __name__ == "__main__":
    # cap = cv2.VideoCapture('videos/destiny_short.mp4')
    cap = cv2.VideoCapture('videos/redbar.mp4')

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


