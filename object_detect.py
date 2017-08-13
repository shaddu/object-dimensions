import imutils
import cv2

image_path = "images/washing_machine.jpg"
image_path2 = "images/A4.jpg"

image = cv2.imread(image_path2)
print image.shape

dim = (525,700)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
print image.shape
cv2.imshow("resized", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey(0)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
cv2.imshow("gray", edged)
cv2.waitKey(0)
edged = cv2.dilate(edged, None, iterations=1)
cv2.imshow("gray", edged)
cv2.waitKey(0)
edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("gray", edged)
cv2.waitKey(0)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

