import numpy as np
import imutils
import cv2

image_path = "images/washing_machine.jpg"
image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)
v.fill(200)
hsv_image = cv2.merge([h,s,v])

rgb_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
cv2.imshow("remove shadows",rgb_image)
gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
# normalize step needs to be added
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# if canny doesnt work then use sobel
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)

cv2.waitKey(0)
