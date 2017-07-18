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
cv2.waitKey(0)

gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
# normalize step needs to be added
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# if canny doesnt work then use sobel
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
cv2.imshow("edged",edged)
cv2.waitKey(0)

#flood fill
h,w = edged.shape[:2]
mask = np.zeros((h+2,w+2),np.uint8)
diff = (6,6,6)
floodfill = cv2.floodFill(edged,mask,(10,10),(0,0,255),diff,diff)

# floodfill = cv2.floodFill(edged,mask, (0,0), 0, cv2.FLOODFILL_MASK_ONLY)

cv2.imshow("flood fill",floodfill[1])
cv2.waitKey(0)
floodfill_mask = (~mask.astype(np.bool))[1:-1, 1:-1]

# Build a mask of the areas inside the face that need inpainting
# inpaint_mask = ~image.mask.mask & floodfill_mask
# cv2.imshow("remove shadows",inpaint_mask)
# cv2.waitKey(0)

edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("erodes",edged)

cv2.waitKey(0)
