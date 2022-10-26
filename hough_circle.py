import numpy as np
import cv2

hough_value = 55
def hough_change(src, gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, 110, hough_value)
    show = src.copy()
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(show, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.imshow("show", show)

src = cv2.imread('11.jpg')
src = cv2.GaussianBlur(src, (3, 3), 0, 0)
cv2.imshow("src", src)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("show")

cv2.createTrackbar("hough_value", "show", hough_value, 200, hough_change)
hough_change(src, gray)

cv2.waitKey(0)