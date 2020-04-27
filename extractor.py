import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
class Extractor(object):

    def __init__(self, K):
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        # self.width, self.height = width, height

    def normalize(self, points):
        return np.dot(self.Kinv, add_ones(points).T).T[:, 0:2]

    def denormalize(self, point):
        ret = np.dot(self.K, np.array([point[0], point[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))

        # return int(round(point[0] + self.width)), int(round(point[1] + self.height))

    def extract(self, image):
        # detection
        features = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

        #extraction
        key_points = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
        key_points, descriptors = self.orb.compute(image, key_points)

        #matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(descriptors, self.last['descriptors'], k=2)
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    key_point1 = key_points[m.queryIdx].pt
                    key_point2 = self.last['key_points'][m.trainIdx].pt
                    ret.append((key_point1, key_point2))
        #    matches = zip([key_points[m.queryIdx] for m in matches], [self.last['key_points'][m.trainIdx] for m in matches])
        
        #normalize coors subtract to move to 0


        # filter
        if len(ret) > 0:
            ret = np.array(ret)
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            # ret[:, :, 0] -= image.shape[0]//2
            # ret[:, :, 1] -= image.shape[1]//2
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)
            ret = ret[inliers]
            # s, v, d = np.linalg.svd(model.params)


        self.last = {'key_points': key_points, 'descriptors': descriptors}


        return ret
