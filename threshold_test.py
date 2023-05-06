import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread("cone3.jpg")
cv.imwrite("test0.png", image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]           # black and white inverted
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)   # black and white
cv.imwrite("test1.png", thresh)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
cv.imwrite("test2.png", opening)

# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

# finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
cv.imwrite("test3.png", unknown)

contours, hierarchies = cv.findContours(unknown, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros(unknown.shape[:2], dtype='uint8')
cv.drawContours(blank, contours, -1, (255, 0, 0), 1)
cv.imwrite("test4.png", blank)

for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv.circle(image, (cx, cy), 7, (0, 0, 255), -1)
        cv.putText(image, "center", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 146, 215), 2)
    # print(f"x: {cx} y: {cy}")

cv.imwrite("test5.png", image)
# cv.waitKey(0)
