from color_thresholding import color_thresh
import numpy as np
from numpy.matlib import repmat
import cv2 as cv
import os
from sklearn.metrics.pairwise import pairwise_distances
from random import shuffle

### Initializing the variables.
dataset_path = './/dataset'
thresh_out_file = 'color_instrument_thresh.npz'
feature_type = 'hu' # 'hu'
num_train_examples = 15
inst_names = {1: 'Retractor', 2: 'Scissors', 3: 'Hemostat', 4: 'Hook', 5: 'Scalpel', 6: 'Gripper', 7: 'Forceps'}
num_instrs = 7
num_hu_moments = 7
binary_thresh = 100
train_flag = True
test_flag = True


### Reading the files
label_file = open(os.path.join(dataset_path, 'labels.txt'),'r')
original_label_lines = label_file.readlines()

## Perform K - Fold
label_lines = original_label_lines[:]
shuffle(label_lines)
train_label_lines = label_lines[:num_train_examples]
test_label_lines = label_lines[num_train_examples:]
print train_label_lines
print test_label_lines

## Color Thresholding - Already Done
# color_thresh(os.path.join(dataset_path, train_label_lines[0].split(' ')[0]), thresh_out_file)

## Opening the saved color thresholding files
npzfiles = np.load(thresh_out_file)
th_min = npzfiles['th_min']
th_max = npzfiles['th_max']

def train(result_filename, train_label_lines):
    avg_features = np.zeros((num_instrs, num_hu_moments))

    # Obtaining the labels
    label_lines = train_label_lines

    for label_line in label_lines:
        print label_line
        words = label_line.split(' ')
        in_file = words[0]
        labels = [int(word) for word in words[1:]]

        frame = cv.imread(os.path.join(dataset_path,in_file))

        mask = cv.inRange(frame, th_min, th_max)
        output = cv.bitwise_and(frame, frame, mask=mask)
        gray_img = cv.cvtColor(output, cv.COLOR_BGR2GRAY)

        _, thresh_img = cv.threshold(gray_img,binary_thresh,255,cv.THRESH_BINARY)

        temp = np.copy(thresh_img)
        temp = cv.morphologyEx(temp, cv.MORPH_CLOSE, np.ones((5,5)), iterations = 3)

        ## Contour extraction
        cont_img, contours, hierarchy = cv.findContours(temp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        area_list = np.array([cv.contourArea(cnt) for cnt in contours])
        peri_list = np.array([cv.arcLength(cnt,True) for cnt in contours])
        arg_list = np.argsort(peri_list+area_list)

        frame_copy = frame.copy()

        features = [[] for _ in range(num_instrs)]

        for idx in range(-1,-1*num_instrs-1,-1):
            cnt = contours[arg_list[idx]]

            ################ Display ################
            ## Draw raw contours
            # cv.drawContours(frame_copy, [cnt], 0, (0,255,0), 5)
            #####
            ### Draw rotated rectangle
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame_copy,[box], 0, (0,255,0), 5)
            cv.putText(frame_copy,inst_names[labels[abs(idx)-1]],(int(rect[0][0]),int(rect[0][1])), cv.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv.LINE_AA)
            #########
            ## Showing the image
            cv.imshow(in_file, frame_copy)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                return;

            # Computing the features
            if feature_type == 'hu':
                Moments = cv.moments(cnt)
                huMoments = cv.HuMoments(Moments).flatten()
                print huMoments
                # feature_vec = [value for key, value in Moments.items() if key[0]=='n']
                feature_vec = huMoments
            elif feature_type == 'surf':
                #### Obtaining the rotated image #####
                img = frame_copy.copy()
                angle = rect[2]
                rows,cols = img.shape[0], img.shape[1]
                M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
                img_rot = cv.warpAffine(img,M,(cols,rows))
                # rotate bounding box
                rect0 = (rect[0], rect[1], 0.0)
                box = cv.boxPoints(rect)
                pts = np.int0(cv.transform(np.array([box]), M))[0]
                pts[pts < 0] = 0
                # crop
                img_crop = img_rot[pts[1][1]:pts[0][1],pts[1][0]:pts[2][0]]
                #########
                surf = cv.xfeatures2d.SURF_create(400)
                kp, des = surf.detectAndCompute(img_crop,None)
                print des.shape()
                feature_vec = np.mean(des,axis=0)
                cv.imshow('Cropped Image', img_crop)
                cv.waitKey(0)


            features[labels[abs(idx)-1]-1] = feature_vec


        avg_features = avg_features + np.array(features)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            return;
        cv.destroyAllWindows()

    avg_features = avg_features / float(len(train_label_lines))
    np.savez(result_filename, avg_features=avg_features)
    print avg_features

def test(result_filename, test_image_path):
    npzfiles = np.load(result_filename)
    avg_features = npzfiles['avg_features']
    mean_vector = np.array([np.mean(abs(avg_features),0)])
    std_vector = np.array([np.mean(abs(avg_features),0)])

    frame = cv.imread(test_image_path)

    mask = cv.inRange(frame, th_min, th_max)
    output = cv.bitwise_and(frame, frame, mask=mask)
    gray_img = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    _, thresh_img = cv.threshold(gray_img,binary_thresh,255,cv.THRESH_BINARY)
    temp = np.copy(thresh_img)
    temp = cv.morphologyEx(temp, cv.MORPH_CLOSE, np.ones((5,5)), iterations = 3)
    ## Contour extraction
    cont_img, contours, hierarchy = cv.findContours(temp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    area_list = np.array([cv.contourArea(cnt) for cnt in contours])
    peri_list = np.array([cv.arcLength(cnt,True) for cnt in contours])
    arg_list = np.argsort(peri_list+area_list)

    frame_copy = frame.copy()
    pred_labels = []

    for idx in range(-1,-1*num_instrs-1,-1):
        cnt = contours[arg_list[idx]]
        Moments = cv.moments(cnt)
        huMoments = cv.HuMoments(Moments).flatten()
        # cn_moments = np.array([value for key, value in Moments.items() if key[0]=='n'])
        cn_moments = huMoments

        # print cn_moments
        # dot_moments = np.sum((avg_features-norm_vector) * (cn_moments-norm_vector),axis=1)
        mod_features = (avg_features-mean_vector)/std_vector
        mod_cn_moments = np.array((cn_moments-mean_vector)/std_vector)
        # print mod_features
        # print mod_cn_moments
        dot_moments = pairwise_distances(mod_features, mod_cn_moments, metric= 'cosine').flatten()
        # print dot_moments
        # pred_labels.append(np.argmax(dot_moments) + 1)
        pred_labels.append(np.argmin(dot_moments) + 1)

        ## Draw raw contours
        # cv.drawContours(frame_copy, [cnt], 0, (0,255,0), 5)
        #####
        ### Draw rotated rectangle
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        # print 'box', box
        box = np.int0(box)
        cv.drawContours(frame_copy,[box], 0, (0,255,0), 5)
        font = cv.FONT_HERSHEY_SIMPLEX
        # print inst_names[pred_labels[-1]]
        # print rect
        cv.putText(frame_copy,inst_names[pred_labels[-1]],(int(rect[0][0]),int(rect[0][1])), font, 2,(255,255,255),2,cv.LINE_AA)
        #########

        # Showing the image
        cv.imshow(test_image_path, frame_copy)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            return;
    return pred_labels


########## Main Code ##########
if train_flag:
    train('result.npz', train_label_lines)
## Testing
if test_flag:
    conf_mat = np.zeros((num_instrs,num_instrs))
    test_files = {line.split(' ')[0]: map(int,line.split(' ')[1:]) for line in test_label_lines}
    # print test_files
    for test_file, true_labels in test_files.items():
        pred_labels = test('result.npz', os.path.join(dataset_path, test_file))
        # print np.array(pred_labels) == np.array(true_labels)
        cv.waitKey(0)
        conf_mat[np.array(true_labels)-1, np.array(pred_labels)-1] = conf_mat[np.array(true_labels)-1, np.array(pred_labels)-1] + 1
    conf_mat = conf_mat / len(test_label_lines)
    print '---- Confusion Matrix ----'
    print conf_mat
    print 'Overall Accuracy', np.mean(conf_mat.diagonal())
