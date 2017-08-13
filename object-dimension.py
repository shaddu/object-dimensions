import numpy as np
import imutils
import cv2
import math
import urllib

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def my_max_function(somelist):
    max_value = None
    for value in somelist:
        if not max_value:
            max_value = value
        elif value > max_value:
            max_value = value
    return max_value


# calculate pixel/metric using reference image
def pixelsPerMetric_finder():
    image = cv2.imread('images/A4.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 10, 80)
    # box2
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # print cnts
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    pixelsPerMetric = None

    contourarray = []
    # append contour area in array and find max out of it
    for i in cnts:
            # print cv2.contourArea(i)
        contourarray.append(float(cv2.contourArea(i)))
        maxcontour = my_max_function(contourarray)

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        # print cv2.contourArea(c)
        if cv2.contourArea(c) < maxcontour:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # loop over the original points and draw them
        # for (x, y) in box:
        #     cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        # dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        dA = math.hypot(tltrX - blbrX, tltrY - blbrY)
        dB = math.hypot(tlblX - trbrX, tlblY - trbrY)
        

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            # to caculate pixelpermetric use the reference object known width
            pixelsPerMetric = dB / 11.6 #6.5
            # print pixelsPerMetric
        return pixelsPerMetric

# Now let's calculate the size of target image
def obj_dimensions(ppm):# image_url="http://answers.opencv.org/upfiles/logo_2.png"):
    # image = io.imread(image_url)
    # image = url_to_image(image_url)
    image = cv2.imread('images/washing_machine.jpg')
    print image.shape

    dim = (525,700)
    # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    print image.shape
    # cv2.imshow("resized", image)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 10, 80)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # print cnts
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # print cnts
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    # (cnts, _) = contours.sort_contours(cnts)

    contourarray = []
    # append contour area in array and find max out of it
    for i in cnts:
        # print cv2.contourArea(i)
        contourarray.append(float(cv2.contourArea(i)))
        maxcontour = my_max_function(contourarray)
    # print contourarray
    # print maxcontour
    # loop over the contours individually
    for c in cnts:

        # print cv2.contourArea(c)
        if cv2.contourArea(c) < maxcontour:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")


        # loop over the original points and draw them
        # for (x, y) in box:
        #     cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        # compute the Euclidean distance between the midpoints
        # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        # dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        dA = math.hypot(tltrX - blbrX, tltrY - blbrY)
        dB = math.hypot(tlblX - trbrX, tlblY - trbrY)
        
        # print dA
        # print dB
        # compute the size of the object
        dimA = dA / ppm
        dimB = dB / ppm
        # item = dynamo_write(image_url,dimA,dimB)
        return round(dimA,1), round(dimB,1)

if __name__ == "__main__":
    ppm = pixelsPerMetric_finder()
    h,w = obj_dimensions(ppm)

        