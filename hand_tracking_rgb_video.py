import numpy as np
import cv2 as cv

############# INPUTS ###################

## Change the path in here
rgb_video_path = './/dataset//gesture-4-color.avi' ##### CHANGE PATH ####
# depth_video_path = './/gesture-4-depth.avi' ##### CHANGE PATH ####

## Input Parameters
ignore_first = 100 ## Ignore first 100 frames

########################################

## Hard Coded
hand_thresh_file = './/color_hand_thresh.npz' ## Don't change this line

cap_rgb = cv.VideoCapture(rgb_video_path)
cap_depth = cv.VideoCapture(depth_video_path)
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)

binary_thresh = 100
extra_incr = 10

## Opening the saved color thresholding files
npzfiles = np.load(hand_thresh_file)
th_min = npzfiles['th_min'] + 1
th_max = npzfiles['th_max'] - 1

if cap_rgb.isOpened():
    ret, frame = cap_rgb.read()
    ret_depth, frame_depth = cap_depth.read()

buffer_count = 0
ignore_count = 0
while(cap_rgb.isOpened() and cap_depth.isOpened()):
    # print buffer_count
    ret, frame = cap_rgb.read()
    ret_depth, frame_depth = cap_depth.read()

    ignore_count = ignore_count+1
    if ignore_count >= ignore_first:
        break

while(cap_rgb.isOpened()):
    buffer_count = buffer_count + 1
    # print buffer_count

    try:
        ##
        ret, frame = cap_rgb.read()
        ret_depth, frame_depth = cap_depth.read()

        mod_img = np.copy(frame)
        mod_img_depth = np.copy(frame_depth)

        ##### Backround Subtraction ###########
        fgmask = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, np.ones((3,3)), iterations=1)
        ## Extractign moving bounding box
        cont_img, contours, hierarchy = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        area_list = np.array([cv.contourArea(cnt) for cnt in contours])
        peri_list = np.array([cv.arcLength(cnt,True) for cnt in contours])
        arg_list = np.argsort(peri_list+area_list)
        if len(arg_list) == 0:
            continue
        max_idx = arg_list[-1]
        cnt = contours[max_idx]
        x,y,w,h = cv.boundingRect(cnt)
        ########################

        ###### Working on moving object
        new_frame = np.zeros(frame.shape, dtype=np.uint8)
        new_frame[y-extra_incr:y+h+extra_incr,x-extra_incr:x+w+extra_incr] = frame[y-extra_incr:y+h+extra_incr,x-extra_incr:x+w+extra_incr]
        mask = cv.inRange(new_frame, th_min, th_max)
        output = cv.bitwise_and(new_frame, new_frame, mask=mask)
        gray_img = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        _, thresh_img = cv.threshold(gray_img,binary_thresh,255,cv.THRESH_BINARY)
        temp = np.copy(thresh_img)
        temp = cv.morphologyEx(temp, cv.MORPH_CLOSE, np.ones((3,3)), iterations = 3)
        ## Contour extraction
        _, contours, _ = cv.findContours(temp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        area_list = np.array([cv.contourArea(cnt) for cnt in contours])
        peri_list = np.array([cv.arcLength(cnt,True) for cnt in contours])
        arg_list = np.argsort(peri_list+area_list)
        if len(arg_list) == 0:
            continue
        max_idx = arg_list[-1]
        cnt = contours[max_idx]
        print 'Perimeter: ', cv.arcLength(cnt,True)

        cv.drawContours(mod_img, [cnt], 0, (0,255,0), 2)
        cv.drawContours(mod_img_depth, [cnt], 0, (0,255,0), 2)

        # Finding the moments and centroids
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cv.circle(mod_img,(int(cx),int(cy)),int(2),(0,95,255),2)
        cv.circle(mod_img_depth,(int(cx),int(cy)),int(2),(0,95,255),2)

        # Finding the convex hull
        hull = cv.convexHull(cnt)
        cv.drawContours(mod_img, [hull], -1, (255,0,0), 2, lineType = 8)
        cv.drawContours(mod_img_depth, [hull], -1, (255,0,0), 2, lineType = 8)

        # Drawing the bounding rectangle
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(mod_img,(x,y),(x+w,y+h),(0,255,0),2)
        print 'Rectangular width-height', [w, h]
        print 'Aspect ratio: ', float(w)/h
        print 'Compactness: ', float(cv.contourArea(cnt))/float(w*h)

        # Rotated bounded rectangle
        min_rect = cv.minAreaRect(cnt)
        min_box = np.int0(cv.boxPoints(min_rect))
        cv.drawContours(mod_img,[min_box],0,(0,0,255),2)
        cv.drawContours(mod_img_depth,[min_box],0,(0,0,255),2)
        print 'Rectangular angle: ', min_rect[-1]

        # Drawing a circle
        (x,y),radius = cv.minEnclosingCircle(cnt)
        cv.circle(mod_img,(int(x),int(y)),int(radius),(0,255,0),2)
        cv.circle(mod_img_depth,(int(x),int(y)),int(radius),(0,255,0),2)

        # Fitting an ellipse
        ellipse = cv.fitEllipse(cnt)
        cv.ellipse(mod_img,ellipse,(0,255,255),2)
        cv.ellipse(mod_img_depth,ellipse,(0,255,255),2)
        print 'Orientation of ellipse: ', ellipse[-1]

        cv.imshow('Modified image', mod_img)
        cv.imshow('Modified image Depth', mod_img_depth)
        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

        cv.imshow('Original-Video',fgmask)

        # cv.imwrite('.//Delete//img%d.jpg'%buffer_count, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print('EOF Error Occured. ')
        break

cap_rgb.release()
cap_depth.release()
cv.destroyAllWindows()
