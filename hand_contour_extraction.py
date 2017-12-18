import cv2 as cv
import numpy as np

img = cv.imread('hand.png')
mod_img = np.copy(img)
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Original Hand', img_gray)
cv.waitKey(0)

## Thresholding and Contour extraction
ret, thresh_img = cv.threshold(img_gray, 127, 255, 0)
cont_img2, contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cont_img = np.copy(img)
cv.drawContours(cont_img, contours, 1, (0,255,0), 5)
cv.drawContours(mod_img, contours, 1, (0,255,0), 5)
cv.imshow('Contoured Image', cont_img)
cv.waitKey(0)

cnt = contours[1]

# Finding the moments and centroids
M = cv.moments(cnt)
print '------------- Moments -------------'
print M
print '-----------------------------------\n'

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print '--------- Centroids ---------------'
print cx, cy
print '-----------------------------------'

# Finding the convex hull
hull = cv.convexHull(cnt)
hull_img = np.copy(img)
cv.drawContours(hull_img, [hull], -1, (255,0,0), 5, lineType = 8)
cv.drawContours(mod_img, [hull], -1, (255,0,0), 5, lineType = 8)
# We should put it wihin square brackets, otherwise, it doesn't draw the hull
cv.imshow('Hull Image', hull_img)
cv.waitKey(0)

# Drawing the bounding rectangle
x,y,w,h = cv.boundingRect(cnt)
rect_img = np.copy(img)
cv.rectangle(rect_img,(x,y),(x+w,y+h),(0,255,0),2)
cv.rectangle(mod_img,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow('Bounded Rectangle Image', rect_img)
cv.waitKey(0)

# Rotated bounded rectangle
min_rect_img = np.copy(img)
min_rect = cv.minAreaRect(cnt)
min_box = np.int0(cv.boxPoints(min_rect))
min_rect_img = cv.drawContours(min_rect_img,[min_box],0,(0,0,255),2)
cv.drawContours(mod_img,[min_box],0,(0,0,255),2)
cv.imshow('Minimum Bounded Rectangle Image', min_rect_img)
cv.waitKey(0)

# Drawing a circle
(x,y),radius = cv.minEnclosingCircle(cnt)
circ_img = np.copy(img)
cv.circle(circ_img,(int(x),int(y)),int(radius),(0,255,0),2)
cv.circle(mod_img,(int(x),int(y)),int(radius),(0,255,0),2)
cv.imshow('Circle Image', circ_img)
cv.waitKey(0)

# Fitting an ellipse
ellipse = cv.fitEllipse(cnt)
ell_img = np.copy(img)
cv.ellipse(ell_img,ellipse,(0,255,255),2)
cv.ellipse(mod_img,ellipse,(0,255,255),2)
cv.imshow('Elliptical Image', ell_img)
cv.waitKey(0)

# Drawing a line
lin_img = np.copy(img)
rows,cols = lin_img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(lin_img,(cols-1,righty),(0,lefty),(0,255,0),2)
cv.line(mod_img,(cols-1,righty),(0,lefty),(0,255,0),2)
cv.imshow('Linear fit Image', lin_img)
cv.waitKey(0)

cv.imshow('Orig. Image', mod_img)
cv.waitKey(0)


