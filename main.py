import os
os.environ["PYTORCH_JIT"] = "0"
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from gui import Ui_MainWindow
from model_loaders import *
from detections import object_detection, landmark_detection, calculate_angles
from utils import *
import threading
import time
import cv2

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        MainWindow.keyPressEvent = self.keyPressEvent
        self.landmarks_label.mousePressEvent = self.mousePressEvent
        self.landmarks_label.mouseMoveEvent = self.mouseMoveEvent
        self.landmarks_label.mouseReleaseEvent = self.mouseReleaseEvent
        self.mousse_is_pressed = False
        self.temp_cropp = [0,0,0,0] # up right down left
        self.total_cropp = [0,0,0,0] # up right down left
        self.crop_vertebra_direction = 1
        self.first_slope = True
        def initialization():
            self.predictor = load_object_detector(model_path = 'object_detection/model/', model_name = 'model_final.pth')
            self.model_loaded = False
            self.tabWidget.setEnabled(True) 
        th = threading.Thread(target=initialization)
        th.start()
        self.choose_image_btn.clicked.connect(self.choose_image) 
        self.launch_btn.clicked.connect(self.launchDetection)
        self.arrowR_bbox_btn.clicked.connect(self.next_bbox_image)
        self.arrowL_bbox_btn.clicked.connect(self.previous_bbox_image)
        self.arrowR_landmarks_btn.clicked.connect(self.next_landmarks_image)
        self.arrowL_landmarks_btn.clicked.connect(self.previous_landmarks_image)
        self.relaunch_bbox_btn.clicked.connect(self.relaunch_bbx)
        self.relaunch_landmarks_btn.clicked.connect(self.relaunch_landmarks)
        check_data_directories()
        self.H_padding = 15
        self.V_padding = 4


    # Open a file dialog and set the selected image on the menu label
    def choose_image(self):
        self.image_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)")
        if self.image_name:
            self.image = cv2.imread(self.image_name)
            self.set_image()
            self.isCropped = False
            self.isNewImage = True
            
    # Set self.image on the menu label
    def set_image(self):
        qimage = convert_cv2QImage(self.image)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.menu_image_label.width(), self.menu_image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.menu_image_label.setPixmap(pixmap)
        self.menu_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.launch_btn.setEnabled(True)
        self.arrowR_bbox_btn.setEnabled(False)
        self.arrowL_bbox_btn.setEnabled(False)
        self.arrowR_landmarks_btn.setEnabled(False)
        self.arrowL_landmarks_btn.setEnabled(False)
        self.index_landmarks = -1
        self.index_bbox = -1
        self.bbox_image_label.clear()
        self.bbox_label.clear()
        self.landmarks_image_label.clear()
        self.landmarks_label.clear()
        self.angle_image_label.clear()
        self.num_bbox_label.setText("")
        self.num_landmarks_label.setText("")
        self.cobb_degree_label.setText("?")
        self.severity_label.setText("?")
        self.treatment_label.setText("?")
        self.isCorrected_bbox = False
        self.isCorrected_landmarks = False

    # Launch the pipeline of detection
    def launchDetection(self):
        self.choose_image_btn.setEnabled(False)
        self.launch_btn.setEnabled(False)
        self.relaunch_bbox_btn.setEnabled(False)
        self.relaunch_landmarks_btn.setEnabled(False)
        def detection():
            # write image 
            if self.isNewImage:
                self.filename = str(time.time())
            save_image(self.isCropped, self.isCorrected_bbox, self.isCorrected_landmarks, self.image, self.image_name, self.filename)
            self.launch_bbox_detection()
            self.launch_landmarks_detection()
            self.launch_angle_calculation(None, None)
            self.isNewImage = False  
            self.choose_image_btn.setEnabled(True)
            self.launch_btn.setEnabled(True)
        th = threading.Thread(target=detection)
        th.start()
    
    # Launch the bbox detection
    def launch_bbox_detection(self):
        # Generate patches
        self.patches, self.bounding_boxes, image_bbox = object_detection(
            self.predictor, 
            self.image,
        )
        self.total_cropp_vertebra = []
        for i in range(len(self.patches)):
            self.total_cropp_vertebra.append([0, 0, 0, 0])
        # Set the bbox image on the bbox window label
        qimage_bbox = convert_cv2QImage(image_bbox)
        pixmap = QtGui.QPixmap.fromImage(qimage_bbox)
        pixmap = pixmap.scaled(self.bbox_image_label.width(), self.bbox_image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.bbox_image_label.setPixmap(pixmap)
        self.bbox_image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Set the visualization of the vertebra bbox
        self.next_bbox_image()
        self.arrowR_bbox_btn.setEnabled(True)
        self.arrowL_bbox_btn.setEnabled(False)
        # Generate csv file
        generate_csv(self.isCropped, self.isCorrected_bbox, self.isCorrected_landmarks, self.bounding_boxes, None, None, None, self.filename, self.total_cropp[0], self.total_cropp[3])
    
    # Launch the landmark detection
    def launch_landmarks_detection(self):
        # Predict all landmarks    
        if not self.model_loaded :
            self.model = load_landmarks_detector('landmark_detection/model/', 'model-090.h5')
            self.model_loaded = True
        self.landmarks, image_landmarks = landmark_detection(
            image = self.image,
            patches = self.patches, 
            bounding_boxes = self.bounding_boxes,
            model = self.model
        )
        # Set the landmarks image on the landmarks window label
        qimage_landmarks = convert_cv2QImage(image_landmarks)
        pixmap = QtGui.QPixmap.fromImage(qimage_landmarks)
        pixmap = pixmap.scaled(self.landmarks_image_label.width(), self.landmarks_image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.landmarks_image_label.setPixmap(pixmap)
        self.landmarks_image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Set the visualization of the vertebra landmarks
        self.next_landmarks_image()
        self.arrowR_landmarks_btn.setEnabled(True)
        self.arrowL_landmarks_btn.setEnabled(False)
        # generate csv file
        generate_csv(self.isCropped, self.isCorrected_bbox, self.isCorrected_landmarks, None, self.landmarks, self.image, self.image_name, self.filename, self.total_cropp[0], self.total_cropp[3])
    
    # Launch the angle calculation
    def launch_angle_calculation(self, lower_MT, upper_MT):
        # Calculate cobb angles, upper MT and lower MT
        cobb_angles, self.upper_MT, self.lower_MT, image_cobb = calculate_angles(self.landmarks, self.image, self.bounding_boxes, lower_MT, upper_MT)
        # Set the visualization of the cobb angle
        qimage_cobb = convert_cv2QImage(image_cobb)
        pixmap = QtGui.QPixmap.fromImage(qimage_cobb)
        pixmap = pixmap.scaled(self.angle_image_label.width(), self.angle_image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.angle_image_label.setPixmap(pixmap)
        self.angle_image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Calculate cobb angles
        self.cobb_degree_label.setText(format(cobb_angles[0],'.2f')+'°')
        if cobb_angles[0]<10:
            self.severity_label.setText("Normal")
            self.treatment_label.setText("---------------------------")
        elif 10<=cobb_angles[0]<25:
            self.severity_label.setText("Moyenne")
            self.treatment_label.setText("Contrôle tous les 2 ans")
        elif 25<=cobb_angles[0]<45:
            self.severity_label.setText("Modérée")
            self.treatment_label.setText("Porter une attelle pendant\n16 à 23 heures par jour")
        else :
            self.severity_label.setText("Sévère")
            self.treatment_label.setText("Révision chirurgicale dans\n20-30 ans")
    
    
    # Next bbox image
    def next_bbox_image(self):
        if self.index_bbox < len(self.patches)-1 :
            self.index_bbox += 1
            qimage_bbox = convert_cv2QImage(self.patches[self.index_bbox])
            pixmap = QtGui.QPixmap.fromImage(qimage_bbox)
            pixmap = pixmap.scaled(self.bbox_label.width(), self.bbox_label.height(), QtCore.Qt.KeepAspectRatio)
            self.bbox_label.setPixmap(pixmap)
            self.bbox_label.setAlignment(QtCore.Qt.AlignCenter)
            if len(self.patches)-1 == self.index_bbox :
                self.arrowR_bbox_btn.setEnabled(False)
            if self.index_bbox > 0 :
                self.arrowL_bbox_btn.setEnabled(True)
            self.num_bbox_label.setText("Vertèbre n°"+str(self.index_bbox+1))
    
    # Previous bbox image    
    def previous_bbox_image(self):
        if self.index_bbox > 0 :
            self.index_bbox -= 1
            qimage_bbox = convert_cv2QImage(self.patches[self.index_bbox])
            pixmap = QtGui.QPixmap.fromImage(qimage_bbox)
            pixmap = pixmap.scaled(self.bbox_label.width(), self.bbox_label.height(), QtCore.Qt.KeepAspectRatio)
            self.bbox_label.setPixmap(pixmap)
            self.bbox_label.setAlignment(QtCore.Qt.AlignCenter)
            if self.index_bbox == 0 :
                self.arrowL_bbox_btn.setEnabled(False)
            if self.index_bbox < len(self.patches)-1 :
                self.arrowR_bbox_btn.setEnabled(True)
            self.num_bbox_label.setText("Vertèbre n°"+str(self.index_bbox+1))
    
    # Next landmarks image
    def next_landmarks_image(self):
        if self.index_landmarks < len(self.patches)-1 :
            self.index_landmarks += 1
            self.set_landmarks_label(None)
            if len(self.patches)-1 == self.index_landmarks :
                self.arrowR_landmarks_btn.setEnabled(False)
            if self.index_landmarks > 0 :
                self.arrowL_landmarks_btn.setEnabled(True)
            self.num_landmarks_label.setText("Vertèbre n°"+str(self.index_landmarks+1))
            self.starting_x = (self.landmarks_label.width() - self.landmarks_label.pixmap().width())/2
            self.starting_y = (self.landmarks_label.height() - self.landmarks_label.pixmap().height())/2
            self.scale = self.patches[self.index_landmarks].shape[0] / self.landmarks_label.pixmap().height()
            self.circles_selected = [False, False, False, False]
    
    # Previous landmarks image
    def previous_landmarks_image(self):
        if self.index_landmarks > 0 :
            self.index_landmarks -= 1
            self.set_landmarks_label(None)
            if self.index_landmarks == 0 :
                self.arrowL_landmarks_btn.setEnabled(False)
            if self.index_landmarks < len(self.patches)-1 :
                self.arrowR_landmarks_btn.setEnabled(True)
            self.num_landmarks_label.setText("Vertèbre n°"+str(self.index_landmarks+1))
            self.starting_x = (self.landmarks_label.width() - self.landmarks_label.pixmap().width())/2
            self.starting_y = (self.landmarks_label.height() - self.landmarks_label.pixmap().height())/2
            self.scale = self.patches[self.index_landmarks].shape[0] / self.landmarks_label.pixmap().height()
            self.circles_selected = [False, False, False, False]
    
    # Set the vertebra image with landmarks on the label
    def set_landmarks_label(self, keypoints) :
        image_landmarks = self.patches[self.index_landmarks].copy()
        if keypoints == None :
            self.keypoints = []
            for i in range(4):
                x = self.landmarks[self.index_landmarks][i][0]-self.bounding_boxes[self.index_landmarks][0]
                y = self.landmarks[self.index_landmarks][i][1]-self.bounding_boxes[self.index_landmarks][1]
                self.keypoints.append((int(x),int(y)))
        image_landmarks = cv2.circle(image_landmarks, self.keypoints[0], 2, (255, 0, 0), 10)
        image_landmarks = cv2.circle(image_landmarks, self.keypoints[1], 2, (255, 0, 0), 10)
        image_landmarks = cv2.circle(image_landmarks, self.keypoints[2], 2, (0, 0, 255), 10)
        image_landmarks = cv2.circle(image_landmarks, self.keypoints[3], 2, (0, 0, 255), 10)
        qimage_landmarks = convert_cv2QImage(image_landmarks)
        pixmap = QtGui.QPixmap.fromImage(qimage_landmarks)
        pixmap = pixmap.scaled(self.landmarks_label.width(), self.landmarks_label.height(), QtCore.Qt.KeepAspectRatio)
        self.landmarks_label.setPixmap(pixmap)
        self.landmarks_label.setAlignment(QtCore.Qt.AlignCenter)
    
    # Relaunch the detection with corrected bbx
    def relaunch_bbx(self):
        self.choose_image_btn.setEnabled(False)
        self.launch_btn.setEnabled(False)
        self.relaunch_bbox_btn.setEnabled(False)
        self.relaunch_landmarks_btn.setEnabled(False)
        self.arrowR_landmarks_btn.setEnabled(False)
        self.arrowL_landmarks_btn.setEnabled(False)
        self.index_landmarks = -1
        self.landmarks_image_label.clear()
        self.landmarks_label.clear()
        self.angle_image_label.clear()
        self.num_landmarks_label.setText("")
        self.cobb_degree_label.setText("?")
        self.severity_label.setText("?")
        self.treatment_label.setText("?")
        def thread_relaunch_bbx():
            save_image(self.isCropped, self.isCorrected_bbox, self.isCorrected_landmarks, self.image, self.image_name, self.filename)
            # Save corrected bbox
            generate_csv(self.isCropped, self.isCorrected_bbox, self.isCorrected_landmarks, self.bounding_boxes, None, None, None, self.filename, self.total_cropp[0], self.total_cropp[3])
            # Launch the detection of landmarks
            self.launch_landmarks_detection()
            # Launch angle calculation
            self.launch_angle_calculation(None, None)  
            self.choose_image_btn.setEnabled(True)
            self.launch_btn.setEnabled(True)
            self.arrowR_landmarks_btn.setEnabled(True)
            self.arrowL_landmarks_btn.setEnabled(True)
        th = threading.Thread(target=thread_relaunch_bbx)
        th.start()
    
    # Relaunch the calculation of the cobb angle with corrected landmarks
    def relaunch_landmarks(self):
        self.choose_image_btn.setEnabled(False)
        self.launch_btn.setEnabled(False)
        self.relaunch_bbox_btn.setEnabled(False)
        self.relaunch_landmarks_btn.setEnabled(False)
        self.angle_image_label.clear()
        self.cobb_degree_label.setText("?")
        self.severity_label.setText("?")
        self.treatment_label.setText("?")
        def thread_relaunch_landmarks():
            save_image(self.isCropped, self.isCorrected_bbox, self.isCorrected_landmarks, self.image, self.image_name, self.filename)
            # Save corrected landmarks
            generate_csv(self.isCropped, self.isCorrected_bbox, self.isCorrected_landmarks, None, self.landmarks, self.image, self.image_name, self.filename, self.total_cropp[0], self.total_cropp[3])
            # Launch angle calculation
            self.launch_angle_calculation(None, None)  
            self.choose_image_btn.setEnabled(True)
            self.launch_btn.setEnabled(True)
            self.arrowR_landmarks_btn.setEnabled(True)
            self.arrowL_landmarks_btn.setEnabled(True)
        th = threading.Thread(target=thread_relaunch_landmarks)
        th.start()
        
    # keypress event handler
    def keyPressEvent(self,event):
        if self.menu_window == self.tabWidget.currentWidget() and self.launch_btn.isEnabled() :
            if event.key() == QtCore.Qt.Key_Up:
                self.temp_cropp[0] += 25
                self.isCropped = True
            elif event.key() == QtCore.Qt.Key_Right:
                self.temp_cropp[1] += 25
                self.isCropped = True
            elif event.key() == QtCore.Qt.Key_Down:
                self.temp_cropp[2] += 25
                self.isCropped = True
            elif event.key() == QtCore.Qt.Key_Left:
                self.temp_cropp[3] += 25
                self.isCropped = True
            y = self.temp_cropp[0]
            h = self.image.shape[0] - self.temp_cropp[2] - y
            x = self.temp_cropp[3]
            w = self.image.shape[1] - self.temp_cropp[1] - x
            self.image = self.image[y:y+h, x:x+w]
            for i in range(4):
                self.total_cropp[i] += self.temp_cropp[i]
                self.temp_cropp[i] = 0
            self.set_image()
        elif self.bbox_window == self.tabWidget.currentWidget() and self.launch_btn.isEnabled() :
            bbox = self.bounding_boxes[self.index_bbox]
            if event.key() == QtCore.Qt.Key_Up:
                if not (bbox[1] + self.temp_cropp[0] < 2 and self.crop_vertebra_direction == -1) :
                    self.temp_cropp[0] += 2 * self.crop_vertebra_direction
                    self.isCorrected_bbox = True
                    self.isCorrected_landmarks = False
            elif event.key() == QtCore.Qt.Key_Right:
                if not (bbox[2] - self.temp_cropp[1] > self.image.shape[1] - 2 and self.crop_vertebra_direction == -1) :
                    self.temp_cropp[1] += 2 * self.crop_vertebra_direction
                    self.isCorrected_bbox = True
                    self.isCorrected_landmarks = False
            elif event.key() == QtCore.Qt.Key_Down:
                if not (bbox[3] - self.temp_cropp[2] > self.image.shape[0] - 2 and self.crop_vertebra_direction == -1) :
                    self.temp_cropp[2] += 2 * self.crop_vertebra_direction
                    self.isCorrected_bbox = True
                    self.isCorrected_landmarks = False
            elif event.key() == QtCore.Qt.Key_Left:
                if not (bbox[0] + self.temp_cropp[3] < 2 and self.crop_vertebra_direction == -1) :
                    self.temp_cropp[3] += 2 * self.crop_vertebra_direction
                    self.isCorrected_bbox = True
                    self.isCorrected_landmarks = False
            elif event.key() == QtCore.Qt.Key_Space:
                self.crop_vertebra_direction = -1 * self.crop_vertebra_direction
            
            if self.isCorrected_bbox :
                self.bounding_boxes[self.index_bbox] = [
                                                        bbox[0] + self.temp_cropp[3],
                                                        bbox[1] + self.temp_cropp[0],
                                                        bbox[2] - self.temp_cropp[1],
                                                        bbox[3] - self.temp_cropp[2]
                                                        ]
                for i in range(4):
                    self.total_cropp_vertebra[self.index_bbox][i] += self.temp_cropp[i]
                    self.temp_cropp[i] = 0
                bbox = self.bounding_boxes[self.index_bbox]
                x1 = int(bbox[0])-self.H_padding
                y1 = int(bbox[1])-self.V_padding
                x2 = int(bbox[2])+self.H_padding
                y2 = int(bbox[3])+self.V_padding
                x1 = x1 if x1 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                x2 = x2 if x2 < self.image.shape[1]  else self.image.shape[1]
                y2 = y2 if y2 < self.image.shape[0] else self.image.shape[0]
                self.patches[self.index_bbox] = self.image.copy()[
                                                                    int(y1):int(y2),
                                                                    int(x1):int(x2)
                                                                  ]
                # Update bbox vertebra image
                qbbox = convert_cv2QImage(self.patches[self.index_bbox])
                pixmap = QtGui.QPixmap.fromImage(qbbox)
                pixmap = pixmap.scaled(self.bbox_label.width(), self.bbox_label.height(), QtCore.Qt.KeepAspectRatio)
                self.bbox_label.setPixmap(pixmap)
                self.bbox_label.setAlignment(QtCore.Qt.AlignCenter)
                # Update all bbox image
                bbox_image = self.image.copy()
                for i, box in enumerate(self.bounding_boxes) :
                    bbox = self.bounding_boxes[i]
                    x1 = int(bbox[0])-self.H_padding
                    y1 = int(bbox[1])-self.V_padding
                    x2 = int(bbox[2])+self.H_padding
                    y2 = int(bbox[3])+self.V_padding
                    x1 = x1 if x1 > 0 else 0
                    y1 = y1 if y1 > 0 else 0
                    x2 = x2 if x2 < self.image.shape[1]  else self.image.shape[1]
                    y2 = y2 if y2 < self.image.shape[0] else self.image.shape[0]
                    if i != self.index_bbox :
                        cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    else :
                        cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                qImage_bbox = convert_cv2QImage(bbox_image)
                pixmap_bbox = QtGui.QPixmap.fromImage(qImage_bbox)
                pixmap_bbox = pixmap_bbox.scaled(self.bbox_image_label.width(), self.bbox_image_label.height(), QtCore.Qt.KeepAspectRatio)
                self.bbox_image_label.setPixmap(pixmap_bbox)
                self.bbox_image_label.setAlignment(QtCore.Qt.AlignCenter)
                self.relaunch_bbox_btn.setEnabled(True)
        elif self.cobb_angle_window == self.tabWidget.currentWidget() and self.launch_btn.isEnabled() :
            if event.key() == QtCore.Qt.Key_Space:
                self.first_slope = not self.first_slope
            elif event.key() == QtCore.Qt.Key_Up:
                if self.first_slope :
                    self.landmarks[self.upper_MT][0] = (self.landmarks[self.upper_MT][0][0], self.landmarks[self.upper_MT][0][1]-10)
                    self.landmarks[self.upper_MT][1] = (self.landmarks[self.upper_MT][1][0], self.landmarks[self.upper_MT][1][1]-10)
                else :
                    self.landmarks[self.lower_MT][2] = (self.landmarks[self.lower_MT][2][0], self.landmarks[self.lower_MT][2][1]-10)
                    self.landmarks[self.lower_MT][3] = (self.landmarks[self.lower_MT][3][0], self.landmarks[self.lower_MT][3][1]-10)
            elif event.key() == QtCore.Qt.Key_Right:
                if self.first_slope :
                    self.landmarks[self.upper_MT][0] = (self.landmarks[self.upper_MT][0][0], self.landmarks[self.upper_MT][0][1]-10)
                    self.landmarks[self.upper_MT][1] = (self.landmarks[self.upper_MT][1][0]+10, self.landmarks[self.upper_MT][1][1])
                else :
                    self.landmarks[self.lower_MT][2] = (self.landmarks[self.lower_MT][2][0], self.landmarks[self.lower_MT][2][1]-10)
                    self.landmarks[self.lower_MT][3] = (self.landmarks[self.lower_MT][3][0]+10, self.landmarks[self.lower_MT][3][1])
            elif event.key() == QtCore.Qt.Key_Down:
                if self.first_slope :
                    self.landmarks[self.upper_MT][0] = (self.landmarks[self.upper_MT][0][0], self.landmarks[self.upper_MT][0][1]+10)
                    self.landmarks[self.upper_MT][1] = (self.landmarks[self.upper_MT][1][0], self.landmarks[self.upper_MT][1][1]+10)
                else :
                    self.landmarks[self.lower_MT][2] = (self.landmarks[self.lower_MT][2][0], self.landmarks[self.lower_MT][2][1]+10)
                    self.landmarks[self.lower_MT][3] = (self.landmarks[self.lower_MT][3][0], self.landmarks[self.lower_MT][3][1]+10)
            elif event.key() == QtCore.Qt.Key_Left:
                if self.first_slope :
                    self.landmarks[self.upper_MT][0] = (self.landmarks[self.upper_MT][0][0], self.landmarks[self.upper_MT][0][1]+10)
                    self.landmarks[self.upper_MT][1] = (self.landmarks[self.upper_MT][1][0]-10, self.landmarks[self.upper_MT][1][1])
                else :
                    self.landmarks[self.lower_MT][2] = (self.landmarks[self.lower_MT][2][0], self.landmarks[self.lower_MT][2][1]+10)
                    self.landmarks[self.lower_MT][3] = (self.landmarks[self.lower_MT][3][0]-10, self.landmarks[self.lower_MT][3][1])
            self.launch_angle_calculation(self.lower_MT, self.upper_MT)
            
    # Mouse press event handler
    def mousePressEvent(self, event):
        self.mousse_is_pressed = True
        self.old_keypoints = self.keypoints.copy()
        pos = (event.pos().x()-self.starting_x)*self.scale, (event.pos().y()-self.starting_y)*self.scale
        for i in range(4):
            r = 10
            x = self.keypoints[i][0]
            y = self.keypoints[i][1]
            if ((pos[0] <= x+r/2) and (pos[0] >= x-r/2) and (pos[1] <= y+r/2) and (pos[1] >= y-r/2)) :
                self.circles_selected[i] = True
                self.relaunch_landmarks_btn.setEnabled(True)
                self.isCorrected_landmarks = True
                self.isCorrected_bbox = False
            
    # Mouse move event handler
    def mouseMoveEvent(self, event):
        if self.mousse_is_pressed:
            for i in range(4):
                if self.circles_selected[i] == True :
                    pos = (event.pos().x()-self.starting_x)*self.scale, (event.pos().y()-self.starting_y)*self.scale
                    self.keypoints[i] = (int(pos[0]), int(pos[1]))
                    diff_x = self.keypoints[i][0] - self.old_keypoints[i][0]
                    diff_y = self.keypoints[i][1] - self.old_keypoints[i][1]
                    new_landmark_pos = self.landmarks[self.index_landmarks][i]
                    self.landmarks[self.index_landmarks][i] = (new_landmark_pos[0]+diff_x, new_landmark_pos[1]+diff_y)
                    self.old_keypoints = self.keypoints.copy()
                    self.set_landmarks_label(self.keypoints)
                    image_landmarks = self.image.copy()
                    for i, l in enumerate(self.landmarks):
                        image_landmarks = cv2.circle(image_landmarks, l[0], 2, (255, 0, 0), 10)
                        image_landmarks = cv2.circle(image_landmarks, l[1], 2, (255, 0, 0), 10)
                        image_landmarks = cv2.circle(image_landmarks, l[2], 2, (0, 0, 255, 255), 10)
                        image_landmarks = cv2.circle(image_landmarks, l[3], 2, (0, 0, 255), 10)
                    # Set the landmarks image on the landmarks window label
                    qimage_landmarks = convert_cv2QImage(image_landmarks)
                    pixmap = QtGui.QPixmap.fromImage(qimage_landmarks)
                    pixmap = pixmap.scaled(self.landmarks_image_label.width(), self.landmarks_image_label.height(), QtCore.Qt.KeepAspectRatio)
                    self.landmarks_image_label.setPixmap(pixmap)
                    self.landmarks_image_label.setAlignment(QtCore.Qt.AlignCenter)
            
    # Mouse release event handler
    def mouseReleaseEvent(self, event):
        self.mousse_is_pressed = False
        self.circles_selected = [False, False, False, False]
        
if __name__ == "__main__":
    import sys
    import os
    os.environ["PYTORCH_JIT"] = "0"
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
        
