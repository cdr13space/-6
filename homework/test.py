import numpy as np
import cv2
src = cv2.imread("0.png")
img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow("src1", img_gray)
ret, img_thres = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("src2", img_thres)
conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img_dilate = cv2.dilate(img_thres, conv_kernel)


cv2.imshow("src1", img_dilate)
cv2.waitKey(0)