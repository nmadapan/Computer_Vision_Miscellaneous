import numpy as np
import cv2 as cv

## Inputs.
left_image_path = './/Images//left_1.jpg'
right_image_path = './/Images//right_1.jpg'

## Initialization
bfm = cv.BFMatcher()
surf = cv.xfeatures2d.SURF_create() #nothing
knn_thresh = 0.75
ransac_reproj_thresh = 4.0

def compute_surf(frame):
	# frame is RGB
	# Returns keypoints and descriptors obtained from SURF features
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	(kps, desc) = surf.detectAndCompute(gray,None)
	kps = np.float32([kp.pt for kp in kps])
	return (kps,desc)

# Match Keypoints
def match_kps(kp1, desc1, kp2, desc2, knn_ratio = 0.75, ransac_reproj_thresh = 4.0):
	desc_matcher = cv.DescriptorMatcher_create("BruteForce")
	raw_matches = desc_matcher.knnMatch(desc1, desc2, 2)
	good_matches = []
	error_matches = []

	for knn_pt1, knn_pt2 in raw_matches:
		if (knn_pt1.distance/knn_pt2.distance) < knn_ratio:
			good_matches.append((knn_pt1.queryIdx, knn_pt1.trainIdx))
			error_matches.append(knn_pt1.distance)

	# We need at least four points to estimate homography
	if len(good_matches) > 4:
		# Obtaining pixel coordinates of key points
		pts1 = np.float32([ kp1[query_idx] for query_idx, _ in good_matches])
		pts2 = np.float32([ kp2[train_idx] for _, train_idx in good_matches])
		# Homography estimation
		(H, status_flag) = cv.findHomography(pts1, pts2, cv.RANSAC, ransac_reproj_thresh)

		# Compute error in homography estimation
		test_pts1 = pts1[status_flag.flatten()==1,:]; test_pts1 = np.append(test_pts1,np.ones((test_pts1.shape[0],1)),axis=1)
		test_pts2 = pts2[status_flag.flatten()==1,:];
		temp = np.dot(H,test_pts1.T); temp = temp / temp[-1,:]; temp = temp[:2,:]
		error = np.mean(np.sqrt(np.mean((test_pts2.T - temp)**2,axis=0)))
		return (good_matches, H, status_flag, error)
	else:
		return None

def draw_matches(frame1, kp1, frame2, kp2, matches, status_flags):
	(h1, w1) = frame1.shape[:2]
	(h2, w2) = frame2.shape[:2]
	output = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
	output[0:h1, 0:w1] = frame1
	output[0:h2, w1:] = frame2
	# loop over the matches
	for ((query_idx, train_idx), status) in zip(matches, status_flags):
		if status == 1:
			# Draw only if homography is successful
			pt1 = (int(kp1[query_idx][0]), int(kp1[query_idx][1]))
			pt2 = (int(kp2[train_idx][0]) + w1, int(kp2[train_idx][1]))
			cv.line(output, pt1, pt2, (0, 255, 0), 1)
	return output

frame1 = cv.imread(right_image_path)
frame2 = cv.imread(left_image_path)
fy = 500 / float(frame1.shape[0])
if fy < 1.0:
	frame1 = cv.resize(frame1, None, fx = fy, fy = fy)
fy = 500 / float(frame2.shape[0])
if fy < 1.0:
	frame2 = cv.resize(frame2, None, fx = fy, fy = fy)

(kp1, desc1) = compute_surf(frame1)
(kp2, desc2) = compute_surf(frame2)
M = match_kps(kp1, desc1, kp2, desc2, knn_ratio = 0.75, ransac_reproj_thresh = 4.0)
if M is None:
	print('Less than 4 points to find homography')

# Warping perspective to stitch the images together
(good_matches, H, status_flags, error) = M
print 'No. of inliers: ', np.sum(status_flags.flatten())
print 'No. of outliers: ', np.sum(status_flags.flatten()==0)
print 'Average residual error: ', error
final_shape = (frame1.shape[1] + frame2.shape[1], max(frame1.shape[0],frame2.shape[0]))
result = cv.warpPerspective(frame1, H, final_shape)
cv.imshow('partial result', result)
result[0:frame2.shape[0], 0:frame2.shape[1]] =\
 np.uint8((0.5*np.float32(result[0:frame2.shape[0], 0:frame2.shape[1]]) + 0.5*np.float32(frame2)))

# Show matches
matched_image = draw_matches(frame1, kp1, frame2, kp2, good_matches, status_flags)

cv.imshow('Matches', matched_image)
cv.imshow('Final Result', result)
cv.waitKey(0)
