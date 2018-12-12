# Usage:
#   python detect_corners image_name method sigma rho threshold_percentile border
# Example:
#   python detect_corners book.png rohr 3 6 95 7
#   python detect_corners book.png harris 3 6 95 3
#
# Non-maxima supression is performed in patches of size (2*border+1) by (2*border+1)

import numpy as np
import cv2 as cv
from scipy.ndimage import filters
import sys
import os

## Non-maxima suppression
def nms(A,brd):
    #Perform non-maxima suppression in (2*brd+1) x (2*brd+1) windows.
    #After nms, only local maxima remain.
    #brd can be 1,2,3,4,...
    Alm = np.zeros(A.shape)
    for i in range(brd,A.shape[0]-brd):
        for j in range(brd,A.shape[1]-brd):
            if A[i, j] < A[i - 1, j] or A[i, j] < A[i + 1, j]:
                Alm[i, j] = 0
            elif A[i, j] < A[i - 1, j - 1] or A[i, j] < A[i + 1, j + 1]:
                Alm[i , j] = 0
            elif A[i, j] < A[i, j - 1] or A[i, j] < A[i, j + 1]:
                Alm[i, j] = 0
            elif A[i, j] < A[i - 1, j + 1] or A[i, j] < A[i + 1, j - 1]:
                Alm[i, j] = 0
            else:
                Alm[i, j] = A[i, j]
    return Alm

#gather input aruments
filename, ext = os.path.splitext(sys.argv[1])
method = sys.argv[2]
sigma = float(sys.argv[3])
rho = float(sys.argv[4])
th = float(sys.argv[5])
brd = int(sys.argv[6])
if brd<1:
    brd=1

#read image, convert to grayscale and normalize between 0 and 1 floating point
img = cv.imread(filename+ext)
res = img.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

#derivative of gaussian
dx = filters.gaussian_filter(img,(sigma,sigma),(1,0))
dy = filters.gaussian_filter(img,(sigma,sigma),(0,1))

#gradient smoothing
dx_sq = filters.gaussian_filter(dx*dx,(rho,rho),(0,0))
dy_sq = filters.gaussian_filter(dy*dy,(rho,rho),(0,0))
dxdy = filters.gaussian_filter(dx*dy,(rho,rho),(0,0))

#compute cornerness
det = (dx_sq * dy_sq) - (dxdy**2) #determinant of structure tensor at every pixel
tr = dx_sq + dy_sq  #trace of structure tensor at every pixel
C = np.zeros(img.shape,np.float32)

#detect corners
if method=='harris':
    C[tr>0] = det[tr>0]/tr[tr>0]**2
    Clm = nms(C,brd)
    th = np.percentile(tr,th)
    tmp = np.logical_and(tr>th,Clm!=0)  #Harris
elif method=='rohr':
    C = np.copy(det)
    Clm = nms(C,brd)
    th = np.percentile(det,th)
    tmp = Clm>th  #Rohr

#draw green circles with radius 5 on corner locations
r,c = tmp.nonzero()
for idx,val in enumerate(r):
    cv.circle(res,(c[idx],r[idx]),5,(0,255,0),-1,cv.LINE_AA)
    #print(idx,r[idx],c[idx],Clm[r[idx],c[idx]])

#normalize as integers between 0 and 255 before saving
det = cv.normalize(det, None, 0,255, cv.NORM_MINMAX, cv.CV_8UC1)
tr = cv.normalize(tr, None, 0,255, cv.NORM_MINMAX, cv.CV_8UC1)
C = cv.normalize(C, None, 0,255, cv.NORM_MINMAX, cv.CV_8UC1)
Clm = cv.normalize(Clm, None, 0,255, cv.NORM_MINMAX, cv.CV_8UC1)

#save results
cv.imwrite(filename+'_det'+'_'+method+'.png',det)
cv.imwrite(filename+'_tr'+'_'+method+'.png',tr)
cv.imwrite(filename+'_cornerness'+'_'+method+'.png',C)
cv.imwrite(filename+'_cornerness_lm'+'_'+method+'.png',Clm)
cv.imwrite(filename+'_res'+'_'+method+'.png',res)
