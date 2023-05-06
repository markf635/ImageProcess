from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils as im
import cv2 as cv

image = cv.imread("cone3.jpg")
shifted = cv.pyrMeanShiftFiltering(image, 21, 51)
cv.imwrite("main4_s1_input.png", image)
cv.imshow("Input", image)

gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
cv.imwrite("main4_s2_thresh.png", thresh)
cv.imshow("Thresh", thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros(thresh.shape[:2], dtype='uint8')
cv.drawContours(blank, contours, -1, (255, 0, 0), 1)
cv.imwrite("main4_s3_contours.png", blank)
cv.imshow("Contours", blank)

for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv.circle(image, (cx, cy), 7, (0, 0, 255), -1)
        cv.putText(image, "center", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 146, 215), 2)
    # print(f"x: {cx} y: {cy}")

cv.imwrite("main4_s4_output.png", image)
cv.imshow("Output", image)
