import numpy as np, cv2
from skimage import io

image = io.imread('http://answers.opencv.org/upfiles/logo_2.png')

#fetch from URL
cv2.imshow('lalala',image)

image_path1 = "images/test.jpg"
image_path2 = "images/example_03.png"
img1 = cv2.imread(image_path1, 0)
img2 = cv2.imread(image_path2, 0)
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
vis[:h1, :w1] = img1
vis[:h2, w1:w1+w2] = img2
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
cv2.imwrite('out.png', vis)
cv2.imshow("test", vis)
cv2.waitKey(0)