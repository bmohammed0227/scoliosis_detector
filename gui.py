# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(650, 900)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(False)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 650, 900))
        self.tabWidget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tabWidget.setStyleSheet("background-color:rgb(240, 240, 240)")
        self.tabWidget.setMovable(False)
        self.tabWidget.setObjectName("tabWidget")
        self.menu_window = QtWidgets.QWidget()
        self.menu_window.setObjectName("menu_window")
        self.choose_image_btn = QtWidgets.QPushButton(self.menu_window)
        self.choose_image_btn.setGeometry(QtCore.QRect(450, 370, 121, 61))
        self.choose_image_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.choose_image_btn.setStyleSheet("font-size: 14px")
        self.choose_image_btn.setObjectName("choose_image_btn")
        self.menu_image_label = QtWidgets.QLabel(self.menu_window)
        self.menu_image_label.setGeometry(QtCore.QRect(20, 17, 361, 841))
        self.menu_image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.menu_image_label.setText("")
        self.menu_image_label.setObjectName("menu_image_label")
        self.launch_btn = QtWidgets.QPushButton(self.menu_window)
        self.launch_btn.setEnabled(False)
        self.launch_btn.setGeometry(QtCore.QRect(450, 440, 121, 61))
        self.launch_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.launch_btn.setStyleSheet("font-size: 14px")
        self.launch_btn.setObjectName("launch_btn")
        self.tabWidget.addTab(self.menu_window, "")
        self.bbox_window = QtWidgets.QWidget()
        self.bbox_window.setObjectName("bbox_window")
        self.bbox_image_label = QtWidgets.QLabel(self.bbox_window)
        self.bbox_image_label.setGeometry(QtCore.QRect(20, 17, 361, 841))
        self.bbox_image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.bbox_image_label.setText("")
        self.bbox_image_label.setObjectName("bbox_image_label")
        self.bbox_label = QtWidgets.QLabel(self.bbox_window)
        self.bbox_label.setGeometry(QtCore.QRect(420, 370, 191, 131))
        self.bbox_label.setFrameShape(QtWidgets.QFrame.Box)
        self.bbox_label.setText("")
        self.bbox_label.setObjectName("bbox_label")
        self.arrowR_bbox_btn = QtWidgets.QPushButton(self.bbox_window)
        self.arrowR_bbox_btn.setEnabled(False)
        self.arrowR_bbox_btn.setGeometry(QtCore.QRect(625, 426, 16, 20))
        self.arrowR_bbox_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.arrowR_bbox_btn.setStyleSheet("font-size: 14px;\n"
"font-weight: bold")
        self.arrowR_bbox_btn.setObjectName("arrowR_bbox_btn")
        self.arrowL_bbox_btn = QtWidgets.QPushButton(self.bbox_window)
        self.arrowL_bbox_btn.setEnabled(False)
        self.arrowL_bbox_btn.setGeometry(QtCore.QRect(390, 426, 16, 20))
        self.arrowL_bbox_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.arrowL_bbox_btn.setStyleSheet("font-size: 14px;\n"
"font-weight: bold")
        self.arrowL_bbox_btn.setObjectName("arrowL_bbox_btn")
        self.num_bbox_label = QtWidgets.QLabel(self.bbox_window)
        self.num_bbox_label.setGeometry(QtCore.QRect(420, 340, 191, 20))
        self.num_bbox_label.setStyleSheet("font-size: 14px")
        self.num_bbox_label.setAlignment(QtCore.Qt.AlignCenter)
        self.num_bbox_label.setObjectName("num_bbox_label")
        self.relaunch_bbox_btn = QtWidgets.QPushButton(self.bbox_window)
        self.relaunch_bbox_btn.setEnabled(False)
        self.relaunch_bbox_btn.setGeometry(QtCore.QRect(475, 530, 75, 23))
        self.relaunch_bbox_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.relaunch_bbox_btn.setObjectName("relaunch_bbox_btn")
        self.tabWidget.addTab(self.bbox_window, "")
        self.landmarks_window = QtWidgets.QWidget()
        self.landmarks_window.setObjectName("landmarks_window")
        self.landmarks_image_label = QtWidgets.QLabel(self.landmarks_window)
        self.landmarks_image_label.setGeometry(QtCore.QRect(20, 17, 361, 841))
        self.landmarks_image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.landmarks_image_label.setText("")
        self.landmarks_image_label.setObjectName("landmarks_image_label")
        self.arrowR_landmarks_btn = QtWidgets.QPushButton(self.landmarks_window)
        self.arrowR_landmarks_btn.setEnabled(False)
        self.arrowR_landmarks_btn.setGeometry(QtCore.QRect(625, 426, 16, 20))
        self.arrowR_landmarks_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.arrowR_landmarks_btn.setStyleSheet("font-size: 14px;\n"
"font-weight: bold")
        self.arrowR_landmarks_btn.setObjectName("arrowR_landmarks_btn")
        self.arrowL_landmarks_btn = QtWidgets.QPushButton(self.landmarks_window)
        self.arrowL_landmarks_btn.setEnabled(False)
        self.arrowL_landmarks_btn.setGeometry(QtCore.QRect(390, 426, 16, 20))
        self.arrowL_landmarks_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.arrowL_landmarks_btn.setStyleSheet("font-size: 14px;\n"
"font-weight: bold")
        self.arrowL_landmarks_btn.setObjectName("arrowL_landmarks_btn")
        self.num_landmarks_label = QtWidgets.QLabel(self.landmarks_window)
        self.num_landmarks_label.setGeometry(QtCore.QRect(420, 340, 191, 20))
        self.num_landmarks_label.setStyleSheet("font-size: 14px")
        self.num_landmarks_label.setAlignment(QtCore.Qt.AlignCenter)
        self.num_landmarks_label.setObjectName("num_landmarks_label")
        self.relaunch_landmarks_btn = QtWidgets.QPushButton(self.landmarks_window)
        self.relaunch_landmarks_btn.setEnabled(False)
        self.relaunch_landmarks_btn.setGeometry(QtCore.QRect(475, 530, 75, 23))
        self.relaunch_landmarks_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.relaunch_landmarks_btn.setObjectName("relaunch_landmarks_btn")
        self.tabWidget.addTab(self.landmarks_window, "")
        self.cobb_angle_window = QtWidgets.QWidget()
        self.cobb_angle_window.setObjectName("cobb_angle_window")
        self.angle_image_label = QtWidgets.QLabel(self.cobb_angle_window)
        self.angle_image_label.setGeometry(QtCore.QRect(20, 17, 361, 841))
        self.angle_image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.angle_image_label.setText("")
        self.angle_image_label.setObjectName("angle_image_label")
        self.label = QtWidgets.QLabel(self.cobb_angle_window)
        self.label.setGeometry(QtCore.QRect(386, 200, 251, 20))
        self.label.setStyleSheet("font-size:19px")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.frame = QtWidgets.QFrame(self.cobb_angle_window)
        self.frame.setGeometry(QtCore.QRect(400, 230, 220, 91))
        self.frame.setStyleSheet("")
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 151, 31))
        self.label_2.setStyleSheet("font-size:14px")
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setGeometry(QtCore.QRect(10, 40, 61, 31))
        self.label_7.setStyleSheet("font-size:14px")
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.cobb_degree_label = QtWidgets.QLabel(self.frame)
        self.cobb_degree_label.setGeometry(QtCore.QRect(160, 10, 55, 31))
        self.cobb_degree_label.setStyleSheet("font-size:14px")
        self.cobb_degree_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.cobb_degree_label.setObjectName("cobb_degree_label")
        self.severity_label = QtWidgets.QLabel(self.frame)
        self.severity_label.setGeometry(QtCore.QRect(70, 40, 101, 31))
        self.severity_label.setStyleSheet("font-size:14px")
        self.severity_label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.severity_label.setObjectName("severity_label")
        self.label_3 = QtWidgets.QLabel(self.cobb_angle_window)
        self.label_3.setGeometry(QtCore.QRect(386, 400, 251, 20))
        self.label_3.setStyleSheet("font-size:19px")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.frame_2 = QtWidgets.QFrame(self.cobb_angle_window)
        self.frame_2.setGeometry(QtCore.QRect(400, 430, 220, 91))
        self.frame_2.setStyleSheet("")
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.treatment_label = QtWidgets.QLabel(self.frame_2)
        self.treatment_label.setGeometry(QtCore.QRect(10, 10, 201, 71))
        self.treatment_label.setStyleSheet("font-size:14px")
        self.treatment_label.setAlignment(QtCore.Qt.AlignCenter)
        self.treatment_label.setObjectName("treatment_label")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.cobb_angle_window)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(430, 570, 160, 80))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.radioBtnCobb = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radioBtnCobb.setChecked(True)
        self.radioBtnCobb.setObjectName("radioBtnCobb")
        self.verticalLayout.addWidget(self.radioBtnCobb)
        self.radioBtnFerguson = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radioBtnFerguson.setObjectName("radioBtnFerguson")
        self.verticalLayout.addWidget(self.radioBtnFerguson)
        self.frame.raise_()
        self.angle_image_label.raise_()
        self.label.raise_()
        self.label_3.raise_()
        self.frame_2.raise_()
        self.verticalLayoutWidget.raise_()
        self.tabWidget.addTab(self.cobb_angle_window, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Détecteur de scoliose"))
        self.choose_image_btn.setText(_translate("MainWindow", "Choisir une image"))
        self.launch_btn.setText(_translate("MainWindow", "Lancer la détection"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.menu_window), _translate("MainWindow", "Menu principal"))
        self.arrowR_bbox_btn.setText(_translate("MainWindow", ">"))
        self.arrowL_bbox_btn.setText(_translate("MainWindow", "<"))
        self.num_bbox_label.setText(_translate("MainWindow", "Vertèbre n°1"))
        self.relaunch_bbox_btn.setText(_translate("MainWindow", "Relancer"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.bbox_window), _translate("MainWindow", "Boîtes englobantes"))
        self.arrowR_landmarks_btn.setText(_translate("MainWindow", ">"))
        self.arrowL_landmarks_btn.setText(_translate("MainWindow", "<"))
        self.num_landmarks_label.setText(_translate("MainWindow", "Vertèbre n°1"))
        self.relaunch_landmarks_btn.setText(_translate("MainWindow", "Relancer"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.landmarks_window), _translate("MainWindow", "Points de repère"))
        self.label.setText(_translate("MainWindow", "Information sur la scoliose"))
        self.label_2.setText(_translate("MainWindow", "Degré d\'angle de cobb :"))
        self.label_7.setText(_translate("MainWindow", "Sévérité :"))
        self.cobb_degree_label.setText(_translate("MainWindow", "?"))
        self.severity_label.setText(_translate("MainWindow", "?"))
        self.label_3.setText(_translate("MainWindow", "Traitement"))
        self.treatment_label.setText(_translate("MainWindow", "?"))
        self.radioBtnCobb.setText(_translate("MainWindow", "Angle de Cobb"))
        self.radioBtnFerguson.setText(_translate("MainWindow", "Angle de Ferguson"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.cobb_angle_window), _translate("MainWindow", "Angle de Cobb"))
