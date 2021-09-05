import numpy as np
import cv2

def kMeans(image,K):
    Z1 = image.reshape((-1,1))
    Z1 = np.float32(Z1)
    criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness,labels,center=cv2.kmeans(Z1,K,None,criteria1,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    compactness = center[labels.flatten()]
    compactness = compactness.reshape((image.shape))
    (miinVall, maaxVall, miinLocc, maaxLocc) = cv2.minMaxLoc(compactness)
    image =  cv2.threshold(compactness,int(maaxVall-1),255,cv2.THRESH_BINARY)[1]
    return image 