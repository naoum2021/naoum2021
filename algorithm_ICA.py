import numpy as np
import cv2
from fastICA_with_g2 import ICA
import datetime

cap = cv2.VideoCapture('bicycle.avi')
i = 0
while(i<250):
    ret, frame = cap.read()
    if i == 0:
        bg = frame
    if i == 1:
        bgn = frame    
        #cv2.imwrite('./output/' + 'background'+ str(i) +'.JPG', bg)
    if i == 125:
        bg_obj = frame
        break
    i += 1
cap.release
(m, n, c) = np.shape(bg)
bg_V = bg.reshape(1, m*n*c)
bg_obj_V = bg_obj.reshape(1, m*n*c)
x = np.append(bg_V, bg_obj_V, axis=0)
w = ICA(x)
i = 0
cap = cv2.VideoCapture('bicycle.avi')
while(i < 250):              #cap.isOpened()):
    ret, frame = cap.read()
    frame = frame.reshape(1, m*n*c)
    frame = np.append(bg_V, frame, axis = 0)
    frame = np.dot(w, frame)[1].reshape(m, n, c)
    #cv2.imwrite('./output/' + 'object'+ str(i) +'.JPG', frame)
    cv2.namedWindow('Object detected', cv2.WINDOW_NORMAL)
    cv2.imshow('Object detected', frame)
    i += 1
    if cv2.waitKey(1) & 0XFF == ord('i'):
        break
cap.release    
cv2.destroyAllWindows()
