from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils as im
import cv2 as cv

image = cv.imread("cone3.jpg")
shifted = cv.pyrMeanShiftFiltering(image, 21, 51)
cv.imshow("Input", image)
cv.imwrite("main3_s1_input.png", image)


gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
cv.imshow("Thresh", thresh)
cv.imwrite("main3_s2_thresh.png", thresh)

D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO {} unique segments found".format(len(np.unique(labels)) - 1))

for label in np.unique(labels):
    if label == 0:
        continue
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = im.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)

    ((x, y), r) = cv.minEnclosingCircle(c)
    cv.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv.imshow("Output", image)
cv.imwrite("main3_s3_output.png", image)
cv.waitKey(0)
