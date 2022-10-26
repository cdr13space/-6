import cv2
import numpy as np

def edge_Prewitt(img):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)

    X = cv2.convertScaleAbs(x)
    Y = cv2.convertScaleAbs(y)
    img_out = cv2.addWeighted(X, 0.5, Y, 0.5, 0)
    return img_out

def edge_Roberts(img):
    kernelx = np.array([[-1, 0] ,[0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)

    X = cv2.convertScaleAbs(x)
    Y = cv2.convertScaleAbs(y)
    img_out = cv2.addWeighted(X, 0.5, Y, 0.5, 0)
    return img_out

def edge_Sobel(img):
    dx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    dy = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    img_out = np.hypot(dx, dy)
    img_out = cv2.convertScaleAbs(img_out)
    #img_out = cv2.addWeighted(X, 0.5, Y, 0.5, 0)
    return img_out

def edge_loG(img):
    gaussianBlur = cv2.GaussianBlur(img, (3, 3), 0)
    img_out = cv2.Laplacian(gaussianBlur, cv2.CV_16S, ksize=3)
    img_out = cv2.convertScaleAbs(img_out)
    return img_out

def edge_canny(img):
    img_out = cv2.Canny(img, 45, 90)
    return img_out

if __name__ == "__main__":
    img = cv2.imread('lena.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_thres = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
    img_thres = img_gray

    img_out1 = edge_Prewitt(img_thres)
    img_out2 = edge_Roberts(img_thres)
    img_out3 = edge_Sobel(img_thres)
    img_out4 = edge_loG(img_thres)
    img_out5 = edge_canny(img_thres)
    cv2.imshow('gray', img_gray)
    cv2.imshow('sobel', img_out3)
    cv2.imshow('loG', img_out4)
    cv2.imshow('canny', img_out5)
    cv2.imshow('prewitt', img_out1)
    cv2.imshow('roberts', img_out2)

    cv2.waitKey(0)
#img_prewitt = edge_Prewitt(img_thres)
#img_sobel = edge_Sobel(img_thres)
#img_laplace = edge_laplace(img_thres)
#img_canny = edge_canny(img_thres)


# img = cv2.imread('lena.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', img_gray)
# cv2.waitKey(0)