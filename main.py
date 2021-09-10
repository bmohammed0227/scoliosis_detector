import os
from PyQt5 import QtCore, QtGui, QtWidgets
from gui import Ui_MainWindow
from utils import *
import threading
import cv2
os.environ["PYTORCH_JIT"] = "0"


class CustomQLabel(QtWidgets.QLabel):
    def __init__(self, parentWindow):
        super(CustomQLabel, self).__init__(parentWindow)

        self.mousse_is_pressed = False
        self.old_keypoints = []
        self.keypoints = []
        self.starting_x = 0
        self.starting_y = 0
        self.scale = 0
        self.circles_selected = []

    # Set the vertebra image with landmarks on the label
    def set_landmarks_label(self, keypoints) :
        xrayImage = window.xrayImage
        image_landmarks = xrayImage.patches[window.index_landmarks].copy()
        if keypoints == None :
            self.keypoints = []
            for i in range(4):
                x = xrayImage.landmarks[window.index_landmarks][i][0]-xrayImage.bboxes[window.index_landmarks][0]
                y = xrayImage.landmarks[window.index_landmarks][i][1]-xrayImage.bboxes[window.index_landmarks][1]
                self.keypoints.append((int(x),int(y)))
        image_landmarks = cv2.circle(image_landmarks, self.keypoints[0], 2, (255, 0, 0), 10)
        image_landmarks = cv2.circle(image_landmarks, self.keypoints[1], 2, (255, 0, 0), 10)
        image_landmarks = cv2.circle(image_landmarks, self.keypoints[2], 2, (0, 0, 255), 10)
        image_landmarks = cv2.circle(image_landmarks, self.keypoints[3], 2, (0, 0, 255), 10)
        qimage_landmarks = convert_cv2QImage(image_landmarks)
        pixmap = QtGui.QPixmap.fromImage(qimage_landmarks)
        pixmap = pixmap.scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio)
        self.setPixmap(pixmap)
        self.setAlignment(QtCore.Qt.AlignCenter)

    # Mouse press event handler
    def mousePressEvent(self, event):
        xrayImage = window.xrayImage
        self.mousse_is_pressed = True
        self.old_keypoints = self.keypoints.copy()
        pos = (event.pos().x()-self.starting_x)*self.scale, (event.pos().y()-self.starting_y)*self.scale
        for i in range(4):
            r = 10
            x = self.keypoints[i][0]
            y = self.keypoints[i][1]
            if ((pos[0] <= x+r/2) and (pos[0] >= x-r/2) and (pos[1] <= y+r/2) and (pos[1] >= y-r/2)) :
                self.circles_selected[i] = True
                window.relaunch_landmarks_btn.setEnabled(True)
                xrayImage.isCorrectedLandmarks = True
                xrayImage.isCorrectedBbox = False

    # Mouse move event handler
    def mouseMoveEvent(self, event):
        if self.mousse_is_pressed:
            for i in range(4):
                if self.circles_selected[i] == True :
                    pos = (event.pos().x()-self.starting_x)*self.scale, (event.pos().y()-self.starting_y)*self.scale
                    self.keypoints[i] = (int(pos[0]), int(pos[1]))
                    diff_x = self.keypoints[i][0] - self.old_keypoints[i][0]
                    diff_y = self.keypoints[i][1] - self.old_keypoints[i][1]
                    new_landmark_pos = window.xrayImage.landmarks[window.index_landmarks][i]
                    window.xrayImage.landmarks[window.index_landmarks][i] = (new_landmark_pos[0]+diff_x, new_landmark_pos[1]+diff_y)
                    self.old_keypoints = self.keypoints.copy()
                    self.set_landmarks_label(self.keypoints)
                    image_landmarks = window.xrayImage.image.copy()
                    for i, l in enumerate(window.xrayImage.landmarks):
                        image_landmarks = cv2.circle(image_landmarks, l[0], 2, (255, 0, 0), 10)
                        image_landmarks = cv2.circle(image_landmarks, l[1], 2, (255, 0, 0), 10)
                        image_landmarks = cv2.circle(image_landmarks, l[2], 2, (0, 0, 255, 255), 10)
                        image_landmarks = cv2.circle(image_landmarks, l[3], 2, (0, 0, 255), 10)
                    # Set the landmarks image on the landmarks window label
                    qimage_landmarks = convert_cv2QImage(image_landmarks)
                    pixmap = QtGui.QPixmap.fromImage(qimage_landmarks)
                    pixmap = pixmap.scaled(window.landmarks_image_label.width(), window.landmarks_image_label.height(), QtCore.Qt.KeepAspectRatio)
                    window.landmarks_image_label.setPixmap(pixmap)
                    window.landmarks_image_label.setAlignment(QtCore.Qt.AlignCenter)

    # Mouse release event handler
    def mouseReleaseEvent(self, event):
        self.mousse_is_pressed = False
        self.circles_selected = [False, False, False, False]

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Set up cutsom QLabel for landmarks tab
        self.landmarks_label = CustomQLabel(self.landmarks_window)
        self.landmarks_label.setGeometry(QtCore.QRect(420, 370, 191, 131))
        self.landmarks_label.setFrameShape(QtWidgets.QFrame.Box)
        self.landmarks_label.setText("")
        self.landmarks_label.setObjectName("landmarks_label")
        self.landmarks_label.setEnabled(False)

        QtCore.QMetaObject.connectSlotsByName(self)

        # MainWindow.keyPressEvent = self.keyPressEvent
        self.crop_vertebra_direction = 1
        self.first_slope = True

        self.xrayImage = XRayImage()
        def initialization():
            self.xrayImage.initialize_models()
            self.tabWidget.setEnabled(True) 
        th = threading.Thread(target=initialization)
        th.start()

        self.choose_image_btn.clicked.connect(self.choose_image) 
        self.launch_btn.clicked.connect(self.launch_detection)
        self.arrowR_bbox_btn.clicked.connect(self.next_bbox_image)
        self.arrowL_bbox_btn.clicked.connect(self.previous_bbox_image)
        self.arrowR_landmarks_btn.clicked.connect(self.next_landmarks_image)
        self.arrowL_landmarks_btn.clicked.connect(self.previous_landmarks_image)
        self.relaunch_bbox_btn.clicked.connect(self.relaunch_bbx)
        self.relaunch_landmarks_btn.clicked.connect(self.relaunch_landmarks)
        self.radioBtnCobb.clicked.connect(self.radioBtnClicked)
        self.radioBtnFerguson.clicked.connect(self.radioBtnClicked)
        check_data_directories()
        self.H_padding = 15
        self.V_padding = 4


    # Open a file dialog and set the selected image on the menu label
    def choose_image(self):
        self.image_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)")
        if self.image_name:
            self.xrayImage.set_image(self.image_name)
            self.display_image(self.xrayImage.get_image())
            
    # Set self.image on the menu label
    def display_image(self, image):
        qimage = convert_cv2QImage(image)
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
        self.xrayImage.isCorrectedBbox = False
        self.xrayImage.isCorrectedLandmarks = False

    # Launch the pipeline of detection
    def launch_detection(self):
        self.choose_image_btn.setEnabled(False)
        self.launch_btn.setEnabled(False)
        self.relaunch_bbox_btn.setEnabled(False)
        self.relaunch_landmarks_btn.setEnabled(False)
        def detection():
            # write image
            self.xrayImage.write_image()
            imageBbox = self.xrayImage.detect_vertebra_bbox()
            self.update_bbox_tab(imageBbox)
            imageLandmarks = self.xrayImage.detect_vertebra_landmarks()
            self.update_landmarks_tab(imageLandmarks)
            imageAngle = self.xrayImage.calculate_angles(None, None)
            self.update_angles_tab(imageAngle)

            self.isNewImage = False  
            self.choose_image_btn.setEnabled(True)
            self.launch_btn.setEnabled(True)
        th = threading.Thread(target=detection)
        th.start()
    
    # Launch the bbox detection
    def update_bbox_tab(self, imageBbox):
        # Set the bbox image on the bbox window label
        qimage_bbox = convert_cv2QImage(imageBbox)
        pixmap = QtGui.QPixmap.fromImage(qimage_bbox)
        pixmap = pixmap.scaled(self.bbox_image_label.width(), self.bbox_image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.bbox_image_label.setPixmap(pixmap)
        self.bbox_image_label.setAlignment(QtCore.Qt.AlignCenter)

        # Set the visualization of the vertebra bbox
        self.next_bbox_image()
        self.arrowR_bbox_btn.setEnabled(True)
        self.arrowL_bbox_btn.setEnabled(False)
    
    # Launch the landmark detection
    def update_landmarks_tab(self, imageLandmarks):
        # Set the landmarks image on the landmarks window label
        qimage_landmarks = convert_cv2QImage(imageLandmarks)
        pixmap = QtGui.QPixmap.fromImage(qimage_landmarks)
        pixmap = pixmap.scaled(self.landmarks_image_label.width(), self.landmarks_image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.landmarks_image_label.setPixmap(pixmap)
        self.landmarks_image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Set the visualization of the vertebra landmarks
        self.next_landmarks_image()
        self.landmarks_label.setEnabled(True)
        self.arrowR_landmarks_btn.setEnabled(True)
        self.arrowL_landmarks_btn.setEnabled(False)
    
    # Launch the angle calculation
    def update_angles_tab(self, imageAngle):
        # Set the visualization of the cobb angle
        qimage_cobb = convert_cv2QImage(imageAngle)
        pixmap = QtGui.QPixmap.fromImage(qimage_cobb)
        pixmap = pixmap.scaled(self.angle_image_label.width(), self.angle_image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.angle_image_label.setPixmap(pixmap)
        self.angle_image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Calculate cobb angles
        cobbAngles = self.xrayImage.cobbAngles
        self.cobb_degree_label.setText(format(cobbAngles[0],'.2f')+'°')
        if cobbAngles[0]<10:
            self.severity_label.setText("Normal")
            self.treatment_label.setText("---------------------------")
        elif 10<=cobbAngles[0]<25:
            self.severity_label.setText("Moyenne")
            self.treatment_label.setText("Contrôle tous les 2 ans")
        elif 25<=cobbAngles[0]<45:
            self.severity_label.setText("Modérée")
            self.treatment_label.setText("Porter une attelle pendant\n16 à 23 heures par jour")
        else :
            self.severity_label.setText("Sévère")
            self.treatment_label.setText("Révision chirurgicale dans\n20-30 ans")
    
    
    # Next bbox image
    def next_bbox_image(self):
        if self.index_bbox < len(self.xrayImage.patches)-1 :
            self.index_bbox += 1
            qimage_bbox = convert_cv2QImage(self.xrayImage.patches[self.index_bbox])
            pixmap = QtGui.QPixmap.fromImage(qimage_bbox)
            pixmap = pixmap.scaled(self.bbox_label.width(), self.bbox_label.height(), QtCore.Qt.KeepAspectRatio)
            self.bbox_label.setPixmap(pixmap)
            self.bbox_label.setAlignment(QtCore.Qt.AlignCenter)
            if len(self.xrayImage.patches)-1 == self.index_bbox :
                self.arrowR_bbox_btn.setEnabled(False)
            if self.index_bbox > 0 :
                self.arrowL_bbox_btn.setEnabled(True)
            self.num_bbox_label.setText("Vertèbre n°"+str(self.index_bbox+1))
    
    # Previous bbox image    
    def previous_bbox_image(self):
        if self.index_bbox > 0 :
            self.index_bbox -= 1
            qimage_bbox = convert_cv2QImage(self.xrayImage.patches[self.index_bbox])
            pixmap = QtGui.QPixmap.fromImage(qimage_bbox)
            pixmap = pixmap.scaled(self.bbox_label.width(), self.bbox_label.height(), QtCore.Qt.KeepAspectRatio)
            self.bbox_label.setPixmap(pixmap)
            self.bbox_label.setAlignment(QtCore.Qt.AlignCenter)
            if self.index_bbox == 0 :
                self.arrowL_bbox_btn.setEnabled(False)
            if self.index_bbox < len(self.xrayImage.patches)-1 :
                self.arrowR_bbox_btn.setEnabled(True)
            self.num_bbox_label.setText("Vertèbre n°"+str(self.index_bbox+1))
    
    # Next landmarks image
    def next_landmarks_image(self):
        if self.index_landmarks < len(self.xrayImage.patches)-1 :
            self.index_landmarks += 1
            self.landmarks_label.set_landmarks_label(None)
            if len(self.xrayImage.patches)-1 == self.index_landmarks :
                self.arrowR_landmarks_btn.setEnabled(False)
            if self.index_landmarks > 0 :
                self.arrowL_landmarks_btn.setEnabled(True)
            self.num_landmarks_label.setText("Vertèbre n°"+str(self.index_landmarks+1))
            self.landmarks_label.starting_x = (self.landmarks_label.width() - self.landmarks_label.pixmap().width())/2
            self.landmarks_label.starting_y = (self.landmarks_label.height() - self.landmarks_label.pixmap().height())/2
            self.landmarks_label.scale = self.xrayImage.patches[self.index_landmarks].shape[0] / self.landmarks_label.pixmap().height()
            self.landmarks_label.circles_selected = [False, False, False, False]
    
    # Previous landmarks image
    def previous_landmarks_image(self):
        if self.index_landmarks > 0 :
            self.index_landmarks -= 1
            self.landmarks_label.set_landmarks_label(None)
            if self.index_landmarks == 0 :
                self.arrowL_landmarks_btn.setEnabled(False)
            if self.index_landmarks < len(self.xrayImage.patches)-1 :
                self.arrowR_landmarks_btn.setEnabled(True)
            self.num_landmarks_label.setText("Vertèbre n°"+str(self.index_landmarks+1))
            self.landmarks_label.starting_x = (self.landmarks_label.width() - self.landmarks_label.pixmap().width())/2
            self.landmarks_label.starting_y = (self.landmarks_label.height() - self.landmarks_label.pixmap().height())/2
            self.landmarks_label.scale = self.xrayImage.patches[self.index_landmarks].shape[0] / self.landmarks_label.pixmap().height()
            self.landmarks_label.circles_selected = [False, False, False, False]

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
            self.xrayImage.write_image()
            # Launch the detection of landmarks
            imageLandmarks = self.xrayImage.detect_vertebra_landmarks()
            self.update_landmarks_tab(imageLandmarks)
            # Launch angle calculation
            imageAngle = self.xrayImage.calculate_angles(None, None)
            self.update_angles_tab(imageAngle)

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
            self.xrayImage.write_image()
            self.xrayImage.save_corrected_landmarks()
            # Launch angle calculation
            imageAngle = self.xrayImage.calculate_angles(None, None)
            self.update_angles_tab(imageAngle)

            self.choose_image_btn.setEnabled(True)
            self.launch_btn.setEnabled(True)
            self.arrowR_landmarks_btn.setEnabled(True)
            self.arrowL_landmarks_btn.setEnabled(True)
        th = threading.Thread(target=thread_relaunch_landmarks)
        th.start()

    def radioBtnClicked(self):
        if self.radioBtnCobb.isChecked():
            print("Cobb")
            imageAngle = self.xrayImage.calculate_angles(None, None)
            self.update_angles_tab(imageAngle)
        elif self.radioBtnFerguson.isChecked():
            print("Ferguson")
            imageAngle = self.xrayImage.calculate_angles_ferguson(None, None)
            self.update_angles_tab(imageAngle)

    # keypress event handler
    def keyPressEvent(self,event):
        if self.menu_window == self.tabWidget.currentWidget() and self.launch_btn.isEnabled() :
            if event.key() == QtCore.Qt.Key_Up:
                self.xrayImage.crop_top()
            elif event.key() == QtCore.Qt.Key_Right:
                self.xrayImage.crop_right()
            elif event.key() == QtCore.Qt.Key_Down:
                self.xrayImage.crop_bottom()
            elif event.key() == QtCore.Qt.Key_Left:
                self.xrayImage.crop_left()
            self.xrayImage.apply_crop()
            self.display_image(self.xrayImage.get_image())
        # TODO: replace second condtion by something else from XRayImage class
        elif self.bbox_window == self.tabWidget.currentWidget() and self.launch_btn.isEnabled() :
            bbox = self.xrayImage.bboxes[self.index_bbox]
            if event.key() == QtCore.Qt.Key_Up:
                if not (bbox[1] + self.xrayImage.tempCrop[0] < 2 and self.crop_vertebra_direction == -1) :
                    self.xrayImage.tempCrop[0] += 2 * self.crop_vertebra_direction
                    self.xrayImage.isCorrectedBbox = True
                    self.xrayImage.isCorrectedLandmarks = False
            elif event.key() == QtCore.Qt.Key_Right:
                if not (bbox[2] - self.xrayImage.tempCrop[1] > self.xrayImage.get_shape()[1] - 2 and self.crop_vertebra_direction == -1) :
                    self.xrayImage.tempCrop[1] += 2 * self.crop_vertebra_direction
                    self.xrayImage.isCorrectedBbox = True
                    self.xrayImage.isCorrectedLandmarks = False
            elif event.key() == QtCore.Qt.Key_Down:
                if not (bbox[3] - self.xrayImage.tempCrop[2] > self.xrayImage.get_image()[0] - 2 and self.crop_vertebra_direction == -1) :
                    self.xrayImage.tempCrop[2] += 2 * self.crop_vertebra_direction
                    self.xrayImage.isCorrectedBbox = True
                    self.xrayImage.isCorrectedLandmarks = False
            elif event.key() == QtCore.Qt.Key_Left:
                if not (bbox[0] + self.xrayImage.tempCrop[3] < 2 and self.crop_vertebra_direction == -1) :
                    self.xrayImage.tempCrop[3] += 2 * self.crop_vertebra_direction
                    self.xrayImage.isCorrectedBbox = True
                    self.xrayImage.isCorrectedLandmarks = False
            elif event.key() == QtCore.Qt.Key_Space:
                self.crop_vertebra_direction = -1 * self.crop_vertebra_direction
            
            if self.xrayImage.isCorrectedBbox :
                self.xrayImage.bboxes[self.index_bbox] = [
                                                        bbox[0] + self.xrayImage.tempCrop[3],
                                                        bbox[1] + self.xrayImage.tempCrop[0],
                                                        bbox[2] - self.xrayImage.tempCrop[1],
                                                        bbox[3] - self.xrayImage.tempCrop[2]
                                                        ]
                for i in range(4):
                    self.xrayImage.totalCropVertebra[self.index_bbox][i] += self.xrayImage.tempCrop[i]
                    self.xrayImage.tempCrop[i] = 0
                bbox = self.xrayImage.bboxes[self.index_bbox]
                x1 = int(bbox[0])-self.H_padding
                y1 = int(bbox[1])-self.V_padding
                x2 = int(bbox[2])+self.H_padding
                y2 = int(bbox[3])+self.V_padding
                x1 = x1 if x1 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                x2 = x2 if x2 < self.xrayImage.get_shape()[1]  else self.xrayImage.get_shape()[1]
                y2 = y2 if y2 < self.xrayImage.get_shape()[0] else self.xrayImage.get_shape()[0]
                self.xrayImage.patches[self.index_bbox] = self.xrayImage.image.copy()[
                                                                    int(y1):int(y2),
                                                                    int(x1):int(x2)
                                                                  ]
                # Update bbox vertebra image
                qbbox = convert_cv2QImage(self.xrayImage.patches[self.index_bbox])
                pixmap = QtGui.QPixmap.fromImage(qbbox)
                pixmap = pixmap.scaled(self.bbox_label.width(), self.bbox_label.height(), QtCore.Qt.KeepAspectRatio)
                self.bbox_label.setPixmap(pixmap)
                self.bbox_label.setAlignment(QtCore.Qt.AlignCenter)
                # Update all bbox image
                bbox_image = self.xrayImage.image.copy()
                for i, box in enumerate(self.xrayImage.bboxes) :
                    bbox = self.xrayImage.bboxes[i]
                    x1 = int(bbox[0])-self.H_padding
                    y1 = int(bbox[1])-self.V_padding
                    x2 = int(bbox[2])+self.H_padding
                    y2 = int(bbox[3])+self.V_padding
                    x1 = x1 if x1 > 0 else 0
                    y1 = y1 if y1 > 0 else 0
                    x2 = x2 if x2 < self.xrayImage.get_shape()[1]  else self.xrayImage.get_shape()[1]
                    y2 = y2 if y2 < self.xrayImage.get_shape()[0] else self.xrayImage.get_shape()[0]
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
            upper_MT = self.xrayImage.upper_MT
            lower_MT = self.xrayImage.lower_MT
            if event.key() == QtCore.Qt.Key_Space:
                self.first_slope = not self.first_slope
            elif event.key() == QtCore.Qt.Key_Up:
                if self.first_slope :
                    self.xrayImage.landmarks[upper_MT][0] = (self.xrayImage.landmarks[upper_MT][0][0], self.xrayImage.landmarks[upper_MT][0][1]-10)
                    self.xrayImage.landmarks[upper_MT][1] = (self.xrayImage.landmarks[upper_MT][1][0], self.xrayImage.landmarks[upper_MT][1][1]-10)
                else :
                    self.xrayImage.landmarks[lower_MT][2] = (self.xrayImage.landmarks[lower_MT][2][0], self.xrayImage.landmarks[lower_MT][2][1]-10)
                    self.xrayImage.landmarks[lower_MT][3] = (self.xrayImage.landmarks[lower_MT][3][0], self.xrayImage.landmarks[lower_MT][3][1]-10)
            elif event.key() == QtCore.Qt.Key_Right:
                if self.first_slope :
                    self.xrayImage.landmarks[upper_MT][0] = (self.xrayImage.landmarks[upper_MT][0][0], self.xrayImage.landmarks[upper_MT][0][1]-10)
                    self.xrayImage.landmarks[upper_MT][1] = (self.xrayImage.landmarks[upper_MT][1][0]+10, self.xrayImage.landmarks[upper_MT][1][1])
                else :
                    self.xrayImage.landmarks[lower_MT][2] = (self.xrayImage.landmarks[lower_MT][2][0], self.xrayImage.landmarks[lower_MT][2][1]-10)
                    self.xrayImage.landmarks[lower_MT][3] = (self.xrayImage.landmarks[lower_MT][3][0]+10, self.xrayImage.landmarks[lower_MT][3][1])
            elif event.key() == QtCore.Qt.Key_Down:
                if self.first_slope :
                    self.xrayImage.landmarks[upper_MT][0] = (self.xrayImage.landmarks[upper_MT][0][0], self.xrayImage.landmarks[upper_MT][0][1]+10)
                    self.xrayImage.landmarks[upper_MT][1] = (self.xrayImage.landmarks[upper_MT][1][0], self.xrayImage.landmarks[upper_MT][1][1]+10)
                else :
                    self.xrayImage.landmarks[lower_MT][2] = (self.xrayImage.landmarks[lower_MT][2][0], self.xrayImage.landmarks[lower_MT][2][1]+10)
                    self.xrayImage.landmarks[lower_MT][3] = (self.xrayImage.landmarks[lower_MT][3][0], self.xrayImage.landmarks[lower_MT][3][1]+10)
            elif event.key() == QtCore.Qt.Key_Left:
                if self.first_slope :
                    self.xrayImage.landmarks[upper_MT][0] = (self.xrayImage.landmarks[upper_MT][0][0], self.xrayImage.landmarks[upper_MT][0][1]+10)
                    self.xrayImage.landmarks[upper_MT][1] = (self.xrayImage.landmarks[upper_MT][1][0]-10, self.xrayImage.landmarks[upper_MT][1][1])
                else :
                    self.xrayImage.landmarks[lower_MT][2] = (self.xrayImage.landmarks[lower_MT][2][0], self.xrayImage.landmarks[lower_MT][2][1]+10)
                    self.xrayImage.landmarks[lower_MT][3] = (self.xrayImage.landmarks[lower_MT][3][0]-10, self.xrayImage.landmarks[lower_MT][3][1])
            imageAngle = self.xrayImage.calculate_angles(lower_MT, upper_MT)
            self.update_angles_tab(imageAngle)
            
        
if __name__ == "__main__":
    import sys
    import os
    os.environ["PYTORCH_JIT"] = "0"
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
        
