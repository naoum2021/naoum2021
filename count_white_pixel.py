
import cv2
import numpy as np
  
gt = cv2.imread('./groundtruth/'+'gt'+ str(2259).zfill(6) +'.png')
gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY, dstCn=0)
obj = cv2.imread('./output_YCrCb/'+'obj.jpg')
obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY, dstCn=0)
(m, n) = obj.shape
#print(obj.shape)
x = np.zeros((m,n))
for i in range(1, m):
    for j in range(1, n):
        if obj[i,j] > 1:
            obj[i,j] = 1
            x[i,j] = obj[i,j]
#for i in range(1,m):
    #for j in range(1,n):
        #if gt[i,j] > 200:
            #x[i,j] = 1  
for i in range(1,340):
    for j in range(1,n):
        if gt[i,j] == 0:
            x[i,j] = 0            
#print(gt.shape)
#print(gt.size)
#print(obj.shape)
#print(obj.size)
#s = x.size
tp = 0
for i in range(1, m):
    for j in range(1, n):
        if gt[i,j] == 255:
            if x[i,j] == 1:
                tp = tp + 1
         
# counting the number of white pixels 
white_pix_in_gt = np.sum(gt == 255)   # TP + FN
white_pix_in_obj = x.size - np.sum(x == 0) # TP + FP
# Recall (Re)
Re = tp / white_pix_in_gt
# Precision (Pr)
Pr = tp/ white_pix_in_obj
# F_measure 
F_measure = (2* Pr* Re)/ (Pr+ Re)  

print('Number of white pixels in groundtruth: TP + FN =', white_pix_in_gt)
print('Number of white pixels in object: TP + FP =', white_pix_in_obj)
print('Number of true positives in object: TP =', tp)
print('Recall: Re =', Re)
print('Precision: Pr =', Pr)
print('F_measure =', F_measure)

#cv2.namedWindow('groundtruth', cv2.WINDOW_NORMAL)
#cv2.namedWindow('object', cv2.WINDOW_NORMAL)
cv2.imshow('groundtruth', gt)
cv2.imshow('object', x)
#cv2.imshow('dif1', y)
cv2.waitKey()
cv2.destroyAllWindows()