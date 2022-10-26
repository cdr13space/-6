import numpy as np
import cv2

def dotdetection(src):
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    ret, img_thres = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    conv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_dilate = cv2.dilate(img_thres, conv_kernel)
    nccomps = cv2.connectedComponentsWithStats(img_dilate)
    centroids = nccomps[3]
    for i in range(0, centroids.shape[0]):
        p = tuple(map(int, centroids[i]))
        cv2.putText(img_dilate, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, 8)

    return img_dilate


src1 = cv2.imread("0.png")
#cv2.imshow("src1", src1)
dst1 = dotdetection(src1)
cv2.imshow("dst1", dst1)
src2 = cv2.imread("1.png")
#cv2.imshow("src2", src2)
dst2 = dotdetection(src2)
cv2.imshow("dst", dst2)
src3 = cv2.imread("2.png")
#cv2.imshow("src3", src3)
dst3 = dotdetection(src3)
cv2.imshow("dst3", dst3)
src4 = cv2.imread("3.png")
#cv2.imshow("src4", src4)
dst4 = dotdetection(src4)
cv2.imshow("dst4", dst4)
src5 = cv2.imread("4.png")
#cv2.imshow("src5", src5)
dst5 = dotdetection(src5)
cv2.imshow("dst5", dst5)
src6 = cv2.imread("5.png")
#cv2.imshow("src6", src6)
dst6 = dotdetection(src6)
cv2.imshow("dst6", dst6)
src7 = cv2.imread("6.png")
#cv2.imshow("src7", src7)
dst7 = dotdetection(src7)
cv2.imshow("dst7", dst7)
src8 = cv2.imread("7.png")
#cv2.imshow("src8", src8)
dst8 = dotdetection(src8)
cv2.imshow("dst8", dst8)
src9 = cv2.imread("8.png")
#cv2.imshow("src9", src9)
dst9 = dotdetection(src9)
cv2.imshow("dst9", dst9)
src10 = cv2.imread("9.png")
#cv2.imshow("src10", src10)
dst10 = dotdetection(src10)
cv2.imshow("dst10", dst10)
cv2.waitKey(0)