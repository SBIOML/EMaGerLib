from PyQt6 import QtWidgets
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap
import os
import sys
import threading
import src.utils.gestures_json as gjutils

class RealTimeGestureUi(QWidget):
    labelChanged = pyqtSignal(int)  # Define a signal for changing labels

    def __init__(self, images:list):
        self.app = QApplication([])

        super().__init__()

        self.images_path = images

        # Get the gestures dictionary
        self.images_folder = gjutils.get_images_folder(self.images_path)
        self.gestures_dict = gjutils.get_gestures_dict(self.images_folder)

        self.images_name = ""
        self.img_label = 1
        self.img_index = 0
        self.label_text = "Label Text"
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.setImg(self.img_index))
        

        self.pixmaps = [QPixmap(img) for img in self.images_path]   
        self.pixmaps = [pm.scaled(QSize(400, 400)) for pm in self.pixmaps]
        self.setWindowTitle('RealTime Gesture Recognition')

        layout = QGridLayout()

        self.labelText = QtWidgets.QLabel(self, alignment=Qt.AlignmentFlag.AlignCenter)
        self.labelText.setText(self.label_text)
        layout.addWidget(self.labelText, 0, 0)

        self.gestureImage = QtWidgets.QLabel(self)  # alignment=Qt.AlignCenter
        self.gestureImage.setPixmap(self.pixmaps[self.img_index])
        layout.addWidget(self.gestureImage, 1, 0)

        self.setLayout(layout)

        self.labelChanged.connect(self.setImg)

    

    @pyqtSlot(int)
    def setImg(self, index):
        # Set the image and label
        self.label_text = self.images_name
        self.label_text = f"(label : {self.img_label})   {self.label_text}   [class : {self.img_index}] "
        self.labelText.setText(self.label_text)
        self.gestureImage.setPixmap(self.pixmaps[self.img_index])

    def update_label(self, label:int):
        # Get images path and index
        self.images_name = self.gestures_dict[str(label)]
        self.img_index = gjutils.get_index_from_label(label, self.images_path, self.gestures_dict)
        if self.img_index is None:
            self.img_index = 0
            return
        self.img_label = label
        self.labelChanged.emit(self.img_index)

    def update_index(self, index:int):
        # Get images path and index
        self.img_index = index
        if self.gestures_dict is not None:
            self.img_label = gjutils.get_label_from_index(index, self.images_path, self.gestures_dict)
            self.images_name = self.gestures_dict[str(self.img_label)]
        else:
            self.images_name = self.images_path[index].split("/")[-1].split(".")[0]
        if self.img_label is None:
            self.img_label = -1
            return
        self.labelChanged.emit(self.img_index)

    def run(self):
        self.show()
        self.timer.start(1000)
        self.app.aboutToQuit.connect(self.stop)
        self.app.exec()

    def stop(self):
        self.timer.stop()