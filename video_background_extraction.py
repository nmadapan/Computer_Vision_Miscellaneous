import numpy as np
import cv2 as cv

############# INPUTS ###################

## Change the path in here
video_path = '..//sample_video.avi' ##### CHANGE PATH ####

########################################

cap = cv.VideoCapture(video_path)

ignore_first = 200
buffer_size = 60
update_for_every = 20 # Frames

if cap.isOpened():
    ret, frame = cap.read()

back_image = np.zeros(frame.shape, dtype=np.uint8)
B = np.array(np.zeros((frame.shape[0], frame.shape[1], buffer_size)))
G = np.array(np.zeros((frame.shape[0], frame.shape[1], buffer_size)))
R = np.array(np.zeros((frame.shape[0], frame.shape[1], buffer_size)))

buffer_count = 0
ignore_count = 0
while(cap.isOpened()):
    print buffer_count
    ret, frame = cap.read()
    ignore_count = ignore_count+1
    if ignore_count >= ignore_first:
        B[:,:,buffer_count] = frame[:,:,0]
        G[:,:,buffer_count]= frame[:,:,1]
        R[:,:,buffer_count] = frame[:,:,2]
        buffer_count = buffer_count + 1

    if buffer_count >= buffer_size:
        break

while(cap.isOpened()):
    buffer_count = buffer_count + 1
    print buffer_count
    ret, frame = cap.read()
    if buffer_count%update_for_every == 0:
        b = frame[:,:,0]; g = frame[:,:,1]; r = frame[:,:,2]
        B[:,:,:-1] = B[:,:,1:]; B[:,:,-1] = b
        G[:,:,:-1] = G[:,:,1:]; G[:,:,-1] = g
        R[:,:,:-1] = R[:,:,1:]; R[:,:,-1] = r

        back_image[:,:,0] = np.median(B, axis=2).astype(np.uint8)
        back_image[:,:,1] = np.median(G, axis=2).astype(np.uint8)
        back_image[:,:,2] = np.median(R, axis=2).astype(np.uint8)

    cv.imshow('Original-Video',frame)
    cv.imshow('Background-Video',back_image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
