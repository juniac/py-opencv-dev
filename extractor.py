import cv2
import time
import numpy as np


class Extractor(object):
    GX = 16 // 2
    GY = 12 // 2

    def __init__(self):
        self.orb = cv2.ORB_create(1000)

    def extract(self, image):
        features = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        # print(features)
        return features
