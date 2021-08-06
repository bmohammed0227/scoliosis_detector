from PyQt5.QtGui import QImage
import os
import csv
import numpy as np
import cv2

# Convert cv image to QImage
def convert_cv2QImage(cvImg):
    cvImg = cvImg[:,:,::-1]
    h, w, ch = cvImg.shape
    bytesPerLine = ch * w
    qImg = QImage(cvImg.data.tobytes(), w, h, bytesPerLine, QImage.Format_RGB888)
    return qImg

# Generate the csv files    
def generate_csv(isCropped, isCorrected_bbox, isCorrected_landmarks, bounding_boxes, landmarks, image, image_name, filename, crop_up, crop_left):
    if bounding_boxes is not None :
        if not isCropped :
            if not isCorrected_bbox :
                repertory = 'data/not_cropped/models_predictions/'
            else :
                repertory = 'data/not_cropped/corrections/bbox/'
            write_filename('filenames.csv', filename+'.png', repertory)
            write__bbox('bbox.csv', filename+'.png', repertory, None, None, bounding_boxes)
        else:
            if not isCorrected_bbox :
                repertory = 'data/cropped/models_predictions/'
            else :
                repertory = 'data/cropped/corrections/bbox/'
            write_filename('filenames_cropped.csv', filename+'_cropped.png', repertory)
            write_filename('filenames.csv', filename+'.png', repertory)
            write__bbox('bbox_with_cropping.csv', filename+'_cropped.png', repertory, None, None, bounding_boxes)
            write__bbox('bbox_without_cropping.csv', filename+'.png', repertory, crop_up, crop_left, bounding_boxes)
    elif landmarks is not None :
        if not isCropped :
            if isCorrected_landmarks :
                repertory = 'data/not_cropped/corrections/landmarks/'
                write_filename('filenames.csv', filename+'.png', repertory)
            elif not isCorrected_bbox :
                repertory = 'data/not_cropped/models_predictions/'
            else :
                repertory = 'data/not_cropped/corrections/bbox/'
            write_landmarks('landmarks.csv', repertory, image.shape[1], image.shape[0], crop_up, crop_left, landmarks)
        else:
            if isCorrected_landmarks :
                repertory = 'data/cropped/corrections/landmarks/'
                write_filename('filenames.csv', filename+'.png', repertory)
                write_filename('filenames_cropped.csv', filename+'_cropped.png', repertory)
            elif not isCorrected_bbox :
                repertory = 'data/cropped/models_predictions/'
            else :
                repertory = 'data/cropped/corrections/bbox/'
            write_landmarks('landmarks_with_cropping.csv', repertory, image.shape[1], image.shape[0], None, None, landmarks)
            original_image = cv2.imread(image_name)
            width_original_image = original_image.shape[1]
            height_original_image = original_image.shape[0]
            write_landmarks('landmarks_without_cropping.csv', repertory, width_original_image, height_original_image, crop_up, crop_left, landmarks)

# Write the image name on the csv file        
def write_filename(csv_name, image_name, repository) :
    f = open(repository+csv_name, 'a', newline = '')
    writer = csv.writer(f)
    writer.writerow([image_name])

# Write the bbox on the csv file
def write__bbox(csv_name, image_name, repository, crop_up, crop_left, bounding_boxes):
    f = open(repository+csv_name, 'a', newline = '')
    writer = csv.writer(f)
    if crop_up is not None :
        for bbox in bounding_boxes :
            writer.writerow([image_name, int(bbox[0])+crop_left, int(bbox[1])+crop_up, int(bbox[2])+crop_left, int(bbox[3])+crop_up, 0])
    else :
        for bbox in bounding_boxes :
            writer.writerow([image_name, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 0])
    f.close()

# Write the landmarks on the csv file
def write_landmarks(csv_name, repository, width, height, crop_up, crop_left, landmarks):
    f = open(repository+csv_name, 'a', newline = '')
    landmarks_csv_x = []
    landmarks_csv_y = []
    if crop_up is not None :
        for i in range(len(landmarks)):
            for j in range(4):
                landmarks_csv_x.append((landmarks[i][j][0]+crop_left)/width)
                landmarks_csv_y.append((landmarks[i][j][1]+crop_up)/height)
    else :
        for i in range(len(landmarks)):
            for j in range(4):
                landmarks_csv_x.append(landmarks[i][j][0]/width)
                landmarks_csv_y.append(landmarks[i][j][1]/height)
    writer = csv.writer(f)
    writer.writerow(np.concatenate((landmarks_csv_x, landmarks_csv_y)))
    f.close()

# Save the original image and the cropped image        
def save_image(isCropped, isCorrected_bbox, isCorrected_landmarks, image, image_name, filename):
    if (not isCropped) and (not isCorrected_bbox) and (not isCorrected_landmarks) :
        cv2.imwrite('data/not_cropped/models_predictions/'+filename+'.png', image)
    elif (not isCropped) and isCorrected_bbox :
        cv2.imwrite('data/not_cropped/corrections/bbox/'+filename+'.png', image)
    elif (not isCropped) and isCorrected_landmarks :
        cv2.imwrite('data/not_cropped/corrections/landmarks/'+filename+'.png', image)
    elif isCropped and (not isCorrected_bbox) and (not isCorrected_landmarks) :
        cv2.imwrite('data/cropped/models_predictions/'+filename+'_cropped.png', image)
        old_image = cv2.imread(image_name)
        cv2.imwrite('data/cropped/models_predictions/'+filename+'.png', old_image)
    elif isCropped and isCorrected_bbox :
        cv2.imwrite('data/cropped/corrections/bbox/'+filename+'_cropped.png', image)
        old_image = cv2.imread(image_name)
        cv2.imwrite('data/cropped/corrections/bbox/'+filename+'.png', old_image)
    elif isCropped and isCorrected_landmarks :
        cv2.imwrite('data/cropped/corrections/landmarks/'+filename+'_cropped.png', image)
        old_image = cv2.imread(image_name)
        cv2.imwrite('data/cropped/corrections/landmarks/'+filename+'.png', old_image)

# Check if all the directories and csv files are created, and create all the missing files and directories
def check_data_directories():
    repository = 'data/not_cropped/models_predictions/'
    if not os.path.exists(repository):
        os.makedirs(repository)
        generate_bbox_csv(repository+'bbox.csv')
        generate_simple_csv(repository+'filenames.csv')
        generate_simple_csv(repository+'landmarks.csv')
        
    repository = 'data/not_cropped/corrections/bbox/'
    if not os.path.exists(repository):
        os.makedirs(repository)
        generate_bbox_csv(repository+'bbox.csv')
        generate_simple_csv(repository+'landmarks.csv')
        generate_simple_csv(repository+'filenames.csv')
    
    repository = 'data/not_cropped/corrections/landmarks/'
    if not os.path.exists(repository):
        os.makedirs(repository)
        generate_simple_csv(repository+'landmarks.csv')    
        generate_simple_csv(repository+'filenames.csv')    
 
    repository = 'data/cropped/models_predictions/'
    if not os.path.exists(repository):  
        os.makedirs(repository)
        generate_bbox_csv(repository+'bbox_with_cropping.csv')
        generate_bbox_csv(repository+'bbox_without_cropping.csv')
        generate_simple_csv(repository+'filenames.csv')
        generate_simple_csv(repository+'filenames_cropped.csv')
        generate_simple_csv(repository+'landmarks_with_cropping.csv')
        generate_simple_csv(repository+'landmarks_without_cropping.csv ')
    
    repository = 'data/cropped/corrections/bbox/'
    if not os.path.exists(repository):  
        os.makedirs(repository)
        generate_bbox_csv(repository+'bbox_with_cropping.csv')
        generate_bbox_csv(repository+'bbox_without_cropping.csv')
        generate_simple_csv(repository+'landmarks_with_cropping.csv')
        generate_simple_csv(repository+'landmarks_without_cropping.csv')
        generate_simple_csv(repository+'filenames.csv')
        generate_simple_csv(repository+'filenames_cropped.csv')
    
    repository = 'data/cropped/corrections/landmarks/'
    if not os.path.exists(repository): 
        os.makedirs(repository)
        generate_simple_csv(repository+'landmarks_with_cropping.csv')
        generate_simple_csv(repository+'landmarks_without_cropping.csv')
        generate_simple_csv(repository+'filenames.csv')
        generate_simple_csv(repository+'filenames_cropped.csv')
 
# Generate bbox csv    
def generate_bbox_csv(path):
    f = open(path, 'w', newline = '')
    writer = csv.writer(f)
    writer.writerow(['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])
    f.close()
    
# Generate simple csv (landmarks or filenames csv)
def generate_simple_csv(path):
    f = open(path, 'w', newline = '')
    f.close()