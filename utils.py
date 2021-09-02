from PyQt5.QtGui import QImage
import os
import csv
import numpy as np
import cv2
import time
from model_loaders import *
from detections import object_detection, landmark_detection, calculate_angles

class XRayImage():
    def __init__(self) -> None:
        self.image = np.zeros((1, 1))  # This is just a declaration
        self.imagePath = None
        self.filename = None

        self.isCropped = None
        self.isNewImage = None

        self.tempCrop = [0, 0, 0, 0]
        self.totalCrop = [0, 0, 0, 0]

        # Variables for bounding boxes detection
        self.predictor = None
        self.imageBbox = None
        self.patches = []
        self.bboxes = []
        self.totalCropVertebra = []
        self.isCorrectedBbox = False

        # Variables for landmarks detections
        self.model = None
        self.modelLoaded = None
        self.imageLandmarks = None
        self.landmarks = []
        self.isCorrectedLandmarks = False

        # Variables for angles calculation
        self.imageAngle = None
        self.cobbAngles = []
        self.upper_MT = 0
        self.lower_MT = 0

    def initialize_models(self):
        self.predictor = load_object_detector(model_path = 'object_detection/model/', model_name = 'model_final.pth')
        self.modelLoaded = False

    def set_image(self, imagePath):
        self.image = cv2.imread(imagePath)
        self.imagePath = imagePath
        self.isCropped = False
        self.isNewImage = True
        print("Image " + imagePath + " loaded.")

    def get_image(self):
        return self.image

    def get_shape(self):
        return self.image.shape

    def crop_top(self):
        self.tempCrop[0] += 25
        self.isCropped = True

    def crop_right(self):
        self.tempCrop[1] += 25
        self.isCropped = True

    def crop_bottom(self):
        self.tempCrop[2] += 25
        self.isCropped = True

    def crop_left(self):
        self.tempCrop[3] += 25
        self.isCropped = True

    def apply_crop(self):
        y = self.tempCrop[0]
        h = self.image.shape[0] - self.tempCrop[2] - y
        x = self.tempCrop[3]
        w = self.image.shape[1] - self.tempCrop[1] - x
        self.image = self.image[y:y+h, x:x+w]
        for i in range(4):
            self.totalCrop[i] += self.tempCrop[i]
            self.tempCrop[i] = 0

    # Save the original image and the cropped image
    def write_image(self):
        if self.isNewImage:
            self.filename = str(time.time())
            if (not self.isCropped) and (not self.isCorrectedBbox) and (not self.isCorrectedLandmarks) :
                cv2.imwrite('data/not_cropped/models_predictions/'+self.filename+'.png', self.image)
            elif (not self.isCropped) and self.isCorrectedBbox :
                cv2.imwrite('data/not_cropped/corrections/bbox/'+self.filename+'.png', self.image)
            elif (not self.isCropped) and self.isCorrectedLandmarks :
                cv2.imwrite('data/not_cropped/corrections/landmarks/'+self.filename+'.png', self.image)
            elif self.isCropped and (not self.isCorrectedBbox) and (not self.isCorrectedLandmarks) :
                cv2.imwrite('data/cropped/models_predictions/'+self.filename+'_cropped.png', self.image)
                old_image = cv2.imread(self.imagePath)
                cv2.imwrite('data/cropped/models_predictions/'+self.filename+'.png', old_image)
            elif self.isCropped and self.isCorrectedBbox :
                cv2.imwrite('data/cropped/corrections/bbox/'+self.filename+'_cropped.png', self.image)
                old_image = cv2.imread(self.imagePath)
                cv2.imwrite('data/cropped/corrections/bbox/'+self.filename+'.png', old_image)
            elif self.isCropped and self.isCorrectedLandmarks :
                cv2.imwrite('data/cropped/corrections/landmarks/'+self.filename+'_cropped.png', self.image)
                old_image = cv2.imread(self.imagePath)
                cv2.imwrite('data/cropped/corrections/landmarks/'+self.filename+'.png', old_image)

    def save_corrected_bboxes(self):
        # Save corrected bbox
        generate_csv(self.isCropped, self.isCorrectedBbox, self.isCorrectedLandmarks, self.bboxes, None, None, None, self.filename, self.totalCrop[0], self.totalCrop[3])

    def save_corrected_landmarks(self):
        # Save corrected landmarks
        generate_csv(self.isCropped, self.isCorrectedBbox, self.isCorrectedLandmarks, None, self.landmarks, self.image, self.imagePath, self.filename, self.totalCrop[0], self.totalCrop[3])


    # Launch the bbox detection
    def detect_vertebra_bbox(self):
        # Generate patches
        self.patches, self.bboxes, self.imageBbox = object_detection(
            self.predictor,
            self.image,
        )
        self.totalCropVertebra = []
        for i in range(len(self.patches)):
            self.totalCropVertebra.append([0, 0, 0, 0])

        # Generate csv file
        generate_csv(self.isCropped, self.isCorrectedBbox, self.isCorrectedLandmarks, self.bboxes, None, None, None, self.filename, self.totalCrop[0], self.totalCrop[3])

        return self.imageBbox

    # Launch the landmark detection
    def detect_vertebra_landmarks(self):
        # Predict all landmarks
        if not self.modelLoaded:
            self.model = load_landmarks_detector('landmark_detection/model/', 'model-090.h5')
            self.modelLoaded = True
        self.landmarks, self.imageLandmarks = landmark_detection(
            image = self.image,
            patches = self.patches,
            bounding_boxes = self.bboxes,
            model = self.model
        )

        # generate csv file
        generate_csv(self.isCropped, self.isCorrectedBbox, self.isCorrectedLandmarks, None, self.landmarks, self.image, self.imagePath, self.filename, self.totalCrop[0], self.totalCrop[3])

        return self.imageLandmarks

    # Launch the angle calculation
    def calculate_angles(self, lower_MT, upper_MT):
        # Calculate cobb angles, upper MT and lower MT
        self.cobbAngles, self.upper_MT, self.lower_MT, self.imageAngle = calculate_angles(self.landmarks, self.image, self.bboxes, lower_MT, upper_MT)

        return self.imageAngle

# TODO: make a VertebraImage class and separate its attributes from XRayImage class
class VertebraImage():
    def __init__(self) -> None:
        self.image = None


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