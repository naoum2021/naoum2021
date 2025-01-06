import numpy as np
import cv2
from numpy.core.numeric import empty_like
from fastICA_with_g2 import ICA
import datetime

bg = cv2.imread('./input/'+'in'+ str(1).zfill(6) +'.jpg')
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2YCrCb)
(m, n, c) = np.shape(bg)
#bg_n = cv2.imread('./input/'+'in'+ str(472).zfill(6) +'.jpg')
bg_obj = cv2.imread('./input/'+'in'+ str(440).zfill(6) +'.jpg')

bg_V = bg.reshape(1, m*n*c)
#bg_n_V = bg_n.reshape(1, m*n*c)
bg_obj_V = bg_obj.reshape(1, m*n*c)

#x = np.append(bg_V, bg_n_V, axis=0)
x = np.append(bg_V, bg_obj_V, axis=0)
w = ICA(x)
y = np.dot(w, x) 
#image_bg = y[0].reshape(m, n, c)
image_obj = y[1].reshape(m, n, c)
Y, Cr, Cb = cv2.split(image_obj)
cv2.imshow('Y', Y)
#cv2.imshow('Cr', Cr)
#cv2.imshow('Cb', Cb)
cv2.waitKey()
cv2.destroyAllWindows()
#W = [[0.8356384, 0.54927994], [0.08442008, -0.12843116]]

#cv2.imwrite('./noise_highway/' + 'noise.JPG', image_noise)

#cv2.namedWindow('Object', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Background', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Noise', cv2.WINDOW_NORMAL)
#cv2.imshow('Object', image_obj)
#cv2.imshow('Noise', image_noise)
#cv2.imshow('Background', image_bg)

#bg = cv2.imread("background_B.JPG")
#bg = cv2.imread("frame0.JPG")
#bg_n = cv2.imread("frame5_B.JPG")
#bg_obj = cv2.imread("frame123_M.JPG")
#bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
#bg_obj_gray = cv2.cvtColor(bg_obj, cv2.COLOR_BGR2GRAY)
#im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)


