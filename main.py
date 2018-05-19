import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

image = cv2.imread('./images/blobs.jpg')
image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
#cv2.imshow('original image' , image)
#cv2.waitKey(0)

detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(image)
image_with_keypoints = cv2.drawKeypoints(image , keypoints , np.array([]) , (0 , 0 , 255) , cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#print(keypoints)
#cv2.imshow('all blobs' , image_with_keypoints)
#cv2.imwrite('all_blobs.jpg' , image_with_keypoints)
#cv2.waitKey()
#cv2.destroyAllWindows()

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.9
params.filterByConvexity = False
params.minConvexity = 0.2
params.filterByInertia = True
params.minConvexity = 0.01

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)
image_with_keypoints = cv2.drawKeypoints(image , keypoints , np.array([]) , (0 , 0 , 255) , cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('With red circular blobs' , image_with_keypoints)
#cv2.imwrite('circular blobs.jpg' , image_with_keypoints)
#cv2.waitKey()
#cv2.destroyAllWindows()