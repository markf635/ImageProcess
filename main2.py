import cv2 as cv
import numpy as np

# cam_port = 0
# cam = cv.VideoCapture(cam_port)
# result, image = cam.read()
# if result:
#     cv.imwrite('testCells.png', image)
# else:
#     print("No image")

image = cv.imread("cone3.jpg")
cv.imwrite("main2_s1_input.png", image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)

cv.imwrite("main2_s2_thresh.png", thresh)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros(thresh.shape[:2], dtype='uint8')
cv.drawContours(blank, contours, -1, (255, 0, 0), 1)
cv.imwrite("main2_s3_contours.png", blank)

for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv.circle(image, (cx, cy), 7, (0, 0, 255), -1)
        cv.putText(image, "center", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 146, 215), 2)
    print(f"x: {cx} y: {cy}")

cv.imwrite("main2_s4_output.png", image)