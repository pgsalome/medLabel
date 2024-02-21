import os
import sys
import time
from threading import Thread

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QMouseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QMainWindow,
    QMessageBox,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import ColorMap
from scipy import ndimage
from skimage.transform import resize


class Helper(QObject):
    changed = pyqtSignal(bool)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self.widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, obj, event):
        if obj is self.widget and event.type() == QEvent.Wheel:
            self.changed.emit(event.angleDelta().y() > 0)
            return True

        return super().eventFilter(obj, event)


class MyImageWidget(pg.ImageView):
    def __init__(self, slider, parent=None):
        super().__init__(parent)
        self.slider = slider
        # self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        gv = self.ui.graphicsView
        helper = Helper(gv.viewport())
        helper.changed.connect(self.change_page)

    def change_page(self, state):
        self.jumpFrames(1 if state else -1)

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y() / 8  # Get the scroll delta
        new_value = self.slider.value() + delta // 15
        self.slider.setValue(new_value)


class MedicalImageViewer(QMainWindow):
    wheel_scrolled = pyqtSignal(object, int)

    def __init__(self, image, modality, initial_setup=[True, True, True]):
        super().__init__()

        self.image = image
        self.modality = modality
        self.initial_setup = initial_setup
        self.init_ui()

    #        self.test_thread = TestThread()
    #        self.test_thread.start()

    def init_ui(self):
        widget = QWidget()
        layout = QGridLayout()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.axial_slider = QSlider(Qt.Horizontal)

        # Use the custom ImageView class and pass the slider to the constructor
        self.sagittal_view = MyImageWidget(slider=self.sagittal_slider)
        self.coronal_view = MyImageWidget(slider=self.coronal_slider)
        self.axial_view = MyImageWidget(slider=self.axial_slider)

        self.views = [self.sagittal_view, self.coronal_view, self.axial_view]

        for view in self.views:
            if self.modality == "PT":
                pos = np.array([0.0, 0.5, 1.0])
                color = np.array(
                    [[0, 0, 128, 255], [0, 255, 255, 255], [255, 0, 0, 255]],
                    dtype=np.ubyte,
                )
                cmap = ColorMap(pos, color)
                view.setColorMap(cmap)
            # view.view.setMouseEnabled(x=False, y=False)
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.view.invertY(False)
            view.view.setAspectLocked(True)

            shape_descending = sorted(self.image.shape, reverse=True)
            x, y = shape_descending[:2]
            view.setMinimumSize(x, y)
            #            view.getHistogramWidget().sigLevelsChanged.connect(self.sync_intensity_levels)
            view.installEventFilter(self)

        layout.addWidget(self.sagittal_view, 0, 0)
        layout.addWidget(self.coronal_view, 0, 2)
        layout.addWidget(self.axial_view, 0, 1)

        layout.addWidget(self.sagittal_slider, 1, 0)
        layout.addWidget(self.coronal_slider, 1, 2)
        layout.addWidget(self.axial_slider, 1, 1)

        self.sagittal_slider.setMaximum(self.image.shape[0] - 1)
        self.coronal_slider.setMaximum(self.image.shape[1] - 1)
        self.axial_slider.setMaximum(self.image.shape[2] - 1)

        self.sagittal_slider.setValue(self.image.shape[0] // 2)
        self.coronal_slider.setValue(self.image.shape[1] // 2)
        self.axial_slider.setValue(self.image.shape[2] // 2)

        self.sagittal_slider.valueChanged.connect(self.update_sagittal_slice)
        self.coronal_slider.valueChanged.connect(self.update_coronal_slice)
        self.axial_slider.valueChanged.connect(self.update_axial_slice)

        self.update_sagittal_slice(self.sagittal_slider.value())
        self.update_coronal_slice(self.coronal_slider.value())
        self.update_axial_slice(self.axial_slider.value())

    def sync_intensity_levels(self):
        levels = self.sender().getLevels()

        for view in self.views:
            if view.getHistogramWidget() != self.sender():
                view.getHistogramWidget().setLevels(*levels)

    def update_sagittal_slice(self, value, orientation="sagittal"):
        if orientation != "coronal":
            shape_descending = sorted(self.image.shape, reverse=True)
            x, y = shape_descending[:2]
            image = ndimage.rotate(self.image[value, :, :].T, 90)
            image = resize(
                image, (x, y), order=3, mode="edge", cval=0, anti_aliasing=False
            )
        if self.initial_setup[0]:
            self.sagittal_view.setImage(image, autoRange=True, autoLevels=True)
            self.initial_setup[0] = False
        else:
            self.sagittal_view.setImage(image, autoRange=False, autoLevels=False)

    def update_coronal_slice(self, value, orientation="sagittal"):
        if orientation != "coronal":
            shape_descending = sorted(self.image.shape, reverse=True)
            x, y = shape_descending[:2]
            image = ndimage.rotate(self.image[:, value, :].T, 90)
            image = resize(
                image, (x, y), order=3, mode="edge", cval=0, anti_aliasing=False
            )
        if self.initial_setup[1]:
            self.coronal_view.setImage(image, autoRange=True, autoLevels=True)
            self.initial_setup[1] = False
        else:
            self.coronal_view.setImage(image, autoRange=False, autoLevels=False)

    def update_axial_slice(self, value, orientation="sagittal"):
        if orientation != "axial":
            shape_descending = sorted(self.image.shape, reverse=True)
            x, y = shape_descending[:2]
            image = ndimage.rotate(self.image[:, :, value].T, 90)
            image = resize(
                image, (x, y), order=3, mode="edge", cval=0, anti_aliasing=False
            )
        if self.initial_setup[2]:
            self.axial_view.setImage(image, autoRange=True, autoLevels=True)
            self.initial_setup[2] = False
        else:
            self.axial_view.setImage(image, autoRangFe=False, autoLevels=False)

    def closeEvent(self, event):
        # self.test_thread.stop_thread = True
        event.accept()

    def update_slider_value(self, view, delta):
        new_value = delta // 15
        if view == self.sagittal_view:
            self.sagittal_slider.setValue(self.sagittal_slider.value() + new_value)
        elif view == self.coronal_view:
            self.coronal_slider.setValue(self.coronal_slider.value() + new_value)
        elif view == self.axial_view:
            self.axial_slider.setValue(self.axial_slider.value() + new_value)
