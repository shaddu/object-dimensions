import cv2
import numpy as np

im = cv2.imread("images/example_01.png",1)
h,w = im.shape[:2]

#flood fill example
diff = (6,6,6)
mask = np.zeros((h+2,w+2),np.uint8)
floodfill = cv2.floodFill(im,mask,(10,10),(0,0,255),diff,diff)

plt_im = cv2.cvtColor(floodfill[1],cv2.COLOR_BGR2RGB)
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

while True:
    cv2.imshow("the flood fill image",floodfill[1])
    if cv2.waitKey(10) == 27:
        break

