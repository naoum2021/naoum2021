import numpy as np
import cv2
from fastICA_with_g2 import ICA
import datetime

noise = cv2.imread('./noise_highway/' + 'noise.JPG')
bg = cv2.imread('./input/'+'in'+ str(471).zfill(6) +'.jpg')
(m, n, c) = np.shape(bg)
bg_V = bg.reshape(1, m*n*c)
#w = np.array([[ 0.78643281, 0.61767583], [0.61224092, -0.77951301]])
w = np.array([[ 0.77161807, 0.63608612], [0.54437625, -0.66036742]]) ##
i = str(1).zfill(6)
while(i < str(1200).zfill(6)):              
    frame = cv2.imread('./input/'+'in'+ str(i).zfill(6) +'.jpg')
    #print(frame.shape)
    #(m, n, c) = np.shape(frame)
    frame = frame.reshape(1, m*n*c)
    frame = np.append(bg_V, frame, axis=0)
    frame = np.dot(w, frame)[1].reshape(m, n, c)
    #print(frame[1].shape)
    
    frame = cv2.subtract(frame, noise, dtype= frame.size)
    #frame = cv2.GaussianBlur(frame,(3,3),3.72308327, 3.65276092, 3.73187555, cv2.BORDER_DEFAULT) # for bicycle video
    #frame = cv2.GaussianBlur(frame,(3,3),3.23808222, 3.31307511, 3.82204301, cv2.BORDER_DEFAULT) # for Man01 video
    #ret, frame = cv2.threshold(frame,30,255,cv2.THRESH_BINARY)
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    ########## Erosion + Dilation (befor binarization ) ########
    #kernel = np.ones((2,3), np.uint8)
    #frame = cv2.erode(frame, kernel, iterations= 1)
    #frame = cv2.dilate(frame, kernel, iterations= 5)
    #rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    #frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, rect_kernel)
    #frame = cv2.erode(frame, rect_kernel, iterations= 1)
    ########## binarization ########
    l_min = np.array([10, 10, 10])
    l_max = np.array([255, 255, 255])
    frame = cv2.inRange(frame, l_min, l_max, dst= 255)
    ########## Erosion + Dilation (after binarization ) #########
    #kernel = np.ones((2,3), np.uint8)
    #frame = cv2.erode(frame, kernel, iterations= 1)
    #frame = cv2.dilate(frame, kernel, iterations= 3)
    ########## Opening + Closing #########
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    frame = cv2.erode(frame, kernel, iterations= 1)
    #frame = cv2.dilate(frame, kernel, iterations= 1)
    #frame = cv2.medianBlur(frame, 7, cv2.BORDER_DEFAULT)
        
    cv2.imwrite('./output/' + 'obj'+ str(i).zfill(6) +'.jpg', frame)
    i = str(int(i) + 1).zfill(6) 
    #print(i)
        
#cv2.destroyAllWindows()
