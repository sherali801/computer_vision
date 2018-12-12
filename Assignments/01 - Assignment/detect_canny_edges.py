'''
Usage:
  python detect_canny_edges image_name sigma percentile_high percentile_low
Example:
  python detect_canny_edges book_gray.png 0.2 80 40
'''

import numpy as np
import cv2 as cv
from scipy.ndimage import filters
import sys
import os

#for the recursion in hysteresis thresholding step 
sys.setrecursionlimit(4000)

global e    #edge
global v    #visited
global vcount
global nms
global tl

def check_neighbours(i,j):
    global e    #edge
    global v    #visited
    global vcount
    for k in range(i-1,i+2):
        for l in range(j-1,j+2):
            if k==i and l==j: #do not recheck current pixel
                continue
            if (k<0 or k>=nms.shape[0] or l<0 or l>=nms.shape[1]):   #handle image boundaries
                continue
            elif nms[k,l]>=tl and e[k,l]==0:
                vcount=vcount+1
                v[k,l]=vcount
                e[k,l]=1
                check_neighbours(k,l)
                #cv.imshow("Edges",e)   #uncomment these two lines to visualize
                #cv.waitKey(10)         #how hysterisis thresholding proceeds

#gather input arguments
filename, ext = os.path.splitext(sys.argv[1])
sigma = float(sys.argv[2])
th = float(sys.argv[3])
tl = float(sys.argv[4])

#helper
pi = np.pi

#read image, convert to grayscale and normalize between 0 and 1 floating point
img = cv.imread(filename+'.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.normalize(img.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

#derivative of gaussian
dx = filters.gaussian_filter(img,(sigma,sigma),(1,0))
dy = filters.gaussian_filter(img,(sigma,sigma),(0,1))

#gradient manitude and angle
m = np.sqrt(dx**2+dy**2)
a = np.arctan2(dy,dx)
#print(np.min(a),np.max(a))

#quantization of gradient direction
q = np.zeros(img.shape,np.float32)
q[ np.logical_or(np.logical_and(a>-pi/8, a<=pi/8) , np.logical_or(a>7*pi/8, a<=-7*pi/8) ) ] = 1
q[ np.logical_or(np.logical_and(a>pi/8, a<=3*pi/8) , np.logical_and(a>-7*pi/8, a<=-5*pi/8) ) ] = 2
q[ np.logical_or(np.logical_and(a>3*pi/8, a<=5*pi/8) , np.logical_and(a>-5*pi/8, a<=-3*pi/8) ) ] = 3
q[ np.logical_or(np.logical_and(a>5*pi/8, a<=7*pi/8) , np.logical_and(a>-3*pi/8, a<=-pi/8) ) ] = 4
q[ m<(0.001*np.max(m)) ] = 0    #ignore pixels with very low gradient magnitude
#print(np.min(m),np.max(m))
#print(np.min(a),np.max(a))
#print(np.min(q),np.max(q))

## Non-maxima supression
nms = np.zeros(img.shape,np.float32)
for x in range(1,img.shape[0]-1):
    for y in range(1,img.shape[1]-1):
        if q[x,y]==1:
            if m[x,y]<m[x-1,y] or m[x,y]<=m[x+1,y]:
                nms[x,y]=0
            else:
                nms[x,y]=m[x,y]
        elif q[x,y]==2:
            if m[x,y]<m[x-1,y-1] or m[x,y]<=m[x+1,y+1]:
                nms[x,y]=0
            else:
                nms[x,y]=m[x,y]
        elif q[x,y]==3:
            # ADD CODE HERE
            if m[x,y]<m[x,y-1] or m[x,y]<=m[x,y+1]:
                nms[x,y]=0
            else:
                nms[x,y]=m[x,y]
        elif q[x,y]==4:
            # ADD CODE HERE
            if m[x,y]<m[x+1,y-1] or m[x,y]<=m[x-1,y+1]:
                nms[x,y]=0
            else:
                nms[x,y]=m[x,y]

## Hysteresis Thresholding
th = np.percentile(nms[nms>0],th)
tl = np.percentile(nms[nms>0],tl)
#print(tl,th)
nms_th = nms>=th
nms_tl = nms>=tl
e = np.zeros(img.shape,np.bool_)
v = np.zeros(img.shape,np.int32)
vcount = 0
for x in range(0,e.shape[0]):
    for y in range(0,e.shape[1]):
        if v[x,y]==0:
            vcount=vcount+1
            v[x,y]=vcount
            if nms[x,y]>=th:
                e[x,y]=1
                check_neighbours(x,y)

#save results
m = cv.normalize(m, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
a = cv.normalize(a, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
q = cv.normalize(q, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
nms = cv.normalize(nms, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
nms_th = cv.normalize(nms_th.astype('float'), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
nms_tl = cv.normalize(nms_tl.astype('float'), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
e = cv.normalize(e.astype('int'), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
v = cv.normalize(v, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
#cv.imshow("Edges", np.hstack([img, m, nms, q, e]))
#cv.waitKey(0)
cv.imwrite(filename+'_m.png',m)
cv.imwrite(filename+'_angles.png',a)
cv.imwrite(filename+'_angles_quantized.png',q)
cv.imwrite(filename+'_nms.png',nms)
cv.imwrite(filename+'_nms_th.png',nms_th)
cv.imwrite(filename+'_nms_tl.png',nms_tl)
cv.imwrite(filename+'_e.png',e)
cv.imwrite(filename+'_v.png',v)
