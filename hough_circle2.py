import cv2
import numpy as np
import sys
import math
import copy
def detect_edges(image):
    h = image.shape[0]
    w = image.shape[1]
    sobeling = np.zeros((h, w), np.float64)
    sobelx = [[-3, 0, 3],
              [-10, 0, 10],
              [-3, 0, 3]]
    sobelx = np.array(sobelx)

    sobely = [[-3, -10, -3],
              [0, 0, 0],
              [3, 10, 3]]
    sobely = np.array(sobely)
    gx = 0
    gy = 0
    testi = 0
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            edgex = 0
            edgey = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    edgex += image[k + i, l + j] * sobelx[1 + k, 1 + l]
                    edgey += image[k + i, l + j] * sobely[1 + k, 1 + l]
            gx = abs(edgex)
            gy = abs(edgey)
            sobeling[i, j] = gx + gy
            # if sobeling[i,j]>255:
            #  sobeling[i, j]=255
            # sobeling[i, j] = sobeling[i,j]/255
    return sobeling


def hough_circles(edge_image, edge_thresh, radius_values):
    h = edge_image.shape[0]
    w = edge_image.shape[1]
    # print(h,w)
    edgimg = np.zeros((h, w), np.int64)
    for i in range(h):
        for j in range(w):
            if edge_image[i][j] > edge_thresh:
                edgimg[i][j] = 255
            else:
                edgimg[i][j] = 0

    accum_array = np.zeros((len(radius_values), h, w))
    # return edgimg , []
    for i in range(h):
        print('Hough Transform进度：', i, '/', h)
        for j in range(w):
            if edgimg[i][j] != 0:
                for r in range(len(radius_values)):
                    rr = radius_values[r]
                    hdown = max(0, i - rr)
                    for a in range(hdown, i):
                        b = round(j + math.sqrt(rr * rr - (a - i) * (a - i)))
                        if b >= 0 and b <= w - 1:
                            accum_array[r][a][b] += 1
                            if 2 * i - a >= 0 and 2 * i - a <= h - 1:
                                accum_array[r][2 * i - a][b] += 1
                        if 2 * j - b >= 0 and 2 * j - b <= w - 1:
                            accum_array[r][a][2 * j - b] += 1
                        if 2 * i - a >= 0 and 2 * i - a <= h - 1 and 2 * j - b >= 0 and 2 * j - b <= w - 1:
                            accum_array[r][2 * i - a][2 * j - b] += 1

    return edgimg, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
    returnlist = []
    hlist = []
    wlist = []
    rlist = []
    returnimg = copy.deepcopy(image)
    for r in range(accum_array.shape[0]):
        print('Find Circles 进度：', r, '/', accum_array.shape[0])
        for h in range(accum_array.shape[1]):
            for w in range(accum_array.shape[2]):
                if accum_array[r][h][w] > hough_thresh:

                    tmp = 0
                    for i in range(len(hlist)):
                        if abs(w - wlist[i]) < 10 and abs(h - hlist[i]) < 10:
                            tmp = 1
                            break

                    if tmp == 0:
                        # print(accum_array[r][h][w])
                        rr = radius_values[r]
                        flag = '(h,w,r)is:(' + str(h) + ',' + str(w) + ',' + str(rr) + ')'
                        returnlist.append(flag)
                        hlist.append(h)
                        wlist.append(w)
                        rlist.append(rr)

    print('圆的数量:', len(hlist))

    for i in range(len(hlist)):
        center = (wlist[i], hlist[i])
        rr = rlist[i]

        color = (0, 255, 0)
        thickness = 2
        cv2.circle(returnimg, center, rr, color, thickness)

    return returnlist, returnimg




img = cv2.imread('coin.png')
# print(img.shape[0], img.shape[1])
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # print(gray_image.shape[0], gray_image.shape[1])
img1 = detect_edges(gray_image)
cv2.imshow("img1",img1)
thresh = 1250
# 需要注意的是，在img1中有些地方的像素值是高于255的，这是由于之前的kernel内的数更大
# 但这并不影响图像的显示
# 因此这里的thresh要大于255
radius_values = []
for i in range(10):
    radius_values.append(20 + i)

edgeimg, accum_array = hough_circles(img1, thresh, radius_values)
# cv2.imshow("img2",accum_array)
# Findcircle
hough_thresh = 70
resultlist, resultimg = find_circles(img, accum_array, radius_values, hough_thresh)
cv2.imshow("img3", resultimg)
cv2.waitKey(0)



