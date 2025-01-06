import numpy as np
import cv2
#from numpy.core.fromnumeric import std
#import datetime
import scipy as sc

cap = cv2.VideoCapture('bicycle.avi')
i = 0
w = np.array([[0.72854541, 0.68499751], [-0.68286633, 0.72627874]]) ### 1
#w = np.array([[0.7285454, 0.68499752], [0.68286634, -0.72627874]]) ### 2 (inverse le moin)
#w = np.array([[0.0285312, 0.0214363], [-0.519938, 0.548299]])     #### Fakhfakh
while(i < 10):
    ret, frame = cap.read()
    (m, n, c) = np.shape(frame)
    if i == 0:
        bg = frame
        bg_V = bg.reshape(1, m*n*c)
        #cv2.imwrite('./output/' + 'background'+ str(i) +'.JPG', bg)
    frame = frame.reshape(1, m*n*c)  
    frame = np.append(bg_V, frame, axis = 0)
    frame = np.dot(w, frame)[1].reshape(m, n, c) 
    noise = cv2.meanStdDev(frame)
    #frame = cv2.GaussianBlur(frame, (3,3), 3.23808222, 3.31307511, 3.82204301) # for Man01 video
    frame = cv2.GaussianBlur(frame, (3,3), 3.72308327, 3.65276092, 3.73187555) # for bicycle video
    cv2.namedWindow('Noise detected', cv2.WINDOW_NORMAL)
    cv2.imshow('Noise detected', frame)
    cv2.imwrite('./noise_ICA_BGR/' + 'noise'+ str(i) +'.JPG', frame)
    i += 1
    
    if cv2.waitKey(35) & 0XFF == ord('i'):
        break
print('noise =', noise)
cap.release    
cv2.destroyAllWindows()

        
    
    
    
    
    
    
    