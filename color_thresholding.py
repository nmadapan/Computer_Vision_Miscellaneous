import cv2 as cv
import numpy as np

def nothing(x):
    pass

'''
'image_path' : path to an image
'out_path' : where to write the .npz file containing the thresholds.
'''
def color_thresh(image_path, out_path):
    # image_path = ''

    frame = cv.imread(image_path)
    cv.namedWindow('Original-Image')
    cv.imshow('Original-Image', frame)

    cv.createTrackbar('B_min', 'Original-Image', 0, 255, nothing)
    cv.createTrackbar('G_min', 'Original-Image', 0, 255, nothing)
    cv.createTrackbar('R_min', 'Original-Image', 0, 255, nothing)

    cv.createTrackbar('B_max', 'Original-Image', 0, 255, nothing)
    cv.createTrackbar('G_max', 'Original-Image', 0, 255, nothing)
    cv.createTrackbar('R_max', 'Original-Image', 0, 255, nothing)

    while(1):
        B_min = cv.getTrackbarPos('B_min', 'Original-Image')
        G_min = cv.getTrackbarPos('G_min', 'Original-Image')
        R_min = cv.getTrackbarPos('R_min', 'Original-Image')

        B_max = cv.getTrackbarPos('B_max', 'Original-Image')
        G_max = cv.getTrackbarPos('G_max', 'Original-Image')
        R_max = cv.getTrackbarPos('R_max', 'Original-Image')

        th_min = np.array([B_min, G_min, R_min], dtype='uint8')
        th_max = np.array([B_max, G_max, R_max], dtype='uint8')

        mask = cv.inRange(frame, th_min, th_max)
        output = cv.bitwise_and(frame, frame, mask=mask)

        cv.imshow('New Image', output)
        key = cv.waitKey(1)
        if  key == ord('q'):
            np.savez(out_path, th_min=th_min, th_max=th_max)
            cv.destroyAllWindows()
            return
            # return th_min, th_max

