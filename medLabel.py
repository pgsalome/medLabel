import sys
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QGridLayout, QScrollArea, QLabel
from pyqtgraph import ColorMap
from PyQt5.QtCore import QSize, QEvent, QObject, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import os
import shutil
import nibabel as nib
import nrrd 
import glob
import warnings
import numpy as np
import yaml
from MedicalImageViewer import MedicalImageViewer
warnings.filterwarnings("ignore")
import argparse
from utils import (save_paths_to_csv, process_dicom_folders,create_patient_dict,
get_dicom_folders,delete_small_directories,rearange_files)

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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.ui.normGroup.hide()
        self.ui.roiPlot.hide()

        gv = self.ui.graphicsView
        helper = Helper(gv.viewport())
        helper.changed.connect(self.change_page)

    def change_page(self, state):
        self.jumpFrames(1 if state else -1)


     
class ImageWithButtons(QWidget):
    def __init__(self,index , image, classes, series_description,modality, 
                 button_position, button_callback, dicom_folders,subject_id_position,labels_level1):
        super().__init__()

        self.index = index
        self.image = image
        self.classes = classes
        self.modality = modality
        self.button_position = button_position
        self.series_description = series_description
        self.button_callback = button_callback
        self.dicom_folders = dicom_folders
        self.subject_id_position = subject_id_position
        self.labels_level1 = labels_level1
   
        self.init_ui()

   

    def init_ui(self):
        main_layout = QGridLayout(self)
        # Add series_description as a title
        
        title_label = QLabel(self.series_description)
        main_layout.addWidget(title_label, 0, 1, 1, 2)
        # Create the image plot
        self.image_view =  MyImageWidget()

#        hist_widget = self.image_view.getHistogramWidget()
#        if hist_widget is not None:
#            hist_widget.setParent(None)
#            hist_widget.deleteLater()
#            self.image_view.histogram = None

        # Set the aspect ratio mode to maintain the original aspect ratio
        self.image_view.view.setAspectLocked(True)
        self.image_view.setFixedSize(256, 256) 
        self.image_view.view.setMouseEnabled(x=False, y=True)
        #self.image_view.view.setMouseMode(pg.ViewBox.RectMode)

        # Add double-click event to the image_view
        self.image_view.mouseDoubleClickEvent = self.on_double_click       

        # Set the color map to jet
        if self.modality == "PT":
            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([[0, 0, 128, 255], [0, 255, 255, 255], [255, 0, 0, 255]], dtype=np.ubyte)
            cmap = ColorMap(pos, color)
            self.image_view.setColorMap(cmap)

        #main_layout.addWidget(self.image_view, 1, 0, 1, 2)

        # Add buttons to the left and right side
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        font = QFont()
        font.setPointSize(10)
      

        for i, button_label in enumerate(self.classes):
            
            
            if self.button_position == "left":
                left_button = QPushButton(button_label)
                left_button.setFont(font)
                left_button.setMinimumSize(QSize(60, 25))
                left_layout.addWidget(left_button)
                self.button_position = "right"
                left_button.clicked.connect(lambda checked, index=self.index, image_class=button_label: self.button_callback(index,image_class,self.dicom_folders, self.subject_id_position, self.labels_level1, self.classes))
            else:
                right_button = QPushButton(button_label)
                right_button.setFont(font)
                right_button.setMinimumSize(QSize(60, 25))
                right_layout.addWidget(right_button)
                self.button_position = "left"
                right_button.clicked.connect(lambda checked, index=self.index, image_class=button_label: self.button_callback(index,image_class, self.dicom_folders, self.subject_id_position, self.labels_level1, self.classes))

        main_layout.addLayout(left_layout, 2, 0, 1, 1)
        main_layout.addWidget(self.image_view)
        main_layout.addLayout(right_layout, 2, 2, 1, 1)
        self.image_view.setImage(self.image,autoRange=True, autoLevels=True)

    def on_double_click(self, event):
        #app = QApplication(sys.argv)
        image_path = glob.glob(self.dicom_folders[self.index]+'.nii.gz')
        if image_path:
            image = nib.load(image_path[0]).get_fdata()
        else:
            image_path = glob.glob(self.dicom_folders[self.index]+'.nrrd')
            image,_ = nrrd.read(image_path[0])
        #print(image_path[0])
        viewer = MedicalImageViewer(image,self.modality)
        self.viewer = viewer
        viewer.show()
        


class MainWindow(QMainWindow):
    
    def __init__(self, images, classes, titles, modality, dicom_folders, subject_id_position, labels_level1,button_position="left"):
        super().__init__()
        
       
        self.classes = classes
        self.images = images
        self.button_position = button_position
        self.titles = titles
        self.modalities = modality
        self.dicom_folders = dicom_folders
        self.subject_id_postion = subject_id_position
        self.labels_level1 = labels_level1
        self.init_ui()

    def init_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Create the image grid
        grid_layout = QGridLayout()
        nrows, ncols = len(self.images)//5+1, 4
        #nrows, ncols = , 4
        image_index = 0

        for row in range(nrows):
            for col in range(ncols):
                if image_index < len(self.images):
                    image_with_buttons = ImageWithButtons(image_index,self.images[image_index], self.classes,
                                                          self.titles[image_index],self.modalities[image_index], 
                                                          self.button_position, self.button_pressed, self.dicom_folders,
                                                          self.subject_id_postion,self.labels_level1)
                    image_index += 1
                    grid_layout.addWidget(image_with_buttons, row, col)

        # Add the grid layout to a QWidget and set it as the widget for the scroll area
        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(grid_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        self.setWindowTitle('Image Viewer with Buttons')
        self.show()
#    def wheelEvent(self, event):
#        delta = event.angleDelta().y()
#        scroll_value = delta / 120 * 50  # Set scroll speed to 50 pixels
#        scrollbar = self.centralWidget().verticalScrollBar()
#        scrollbar.setValue(scrollbar.value() - scroll_value)  

    def eventFilter(self, watched, event):
        if event.type() == QEvent.GraphicsSceneWheel:
            return True
        return super().eventFilter(watched, event)        
        
        
    def check_folder_isempty(self,image_path):
        parent_dir = os.path.dirname(image_path)
        directory_contents = os.listdir(parent_dir)
        if not (any(os.path.isdir(os.path.join(parent_dir , entry)) for entry in directory_contents)):
            shutil.rmtree(parent_dir)
            print(f"Empty directory {parent_dir} has been removed.")   

    def update_csv(self,csv_file_path,new_path,image_class):
        root_dir = os.path.dirname(csv_file_path)
        exists = os.path.isfile(csv_file_path)
        check = False
        if exists:
            with open(csv_file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == new_path and row[1] == image_class:
                        print("Image path and class already present in CSV file.")
                        check = True        
        if not check:
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if not exists:
                    writer.writerow(["Image_Path", "Class"])
                writer.writerow([new_path.replace(root_dir,""), image_class])
                 
   #button.on_clicked(create_lambda_func(i, cls, classes,subject_id_position))
         
    def button_pressed(self, i, image_class, dicom_folders, subject_id_position, labels_level1, classes):
        
        #print("ALOHA"+str(i))
        image_path = dicom_folders[i]
        parts = image_path.split(os.sep)
        csv_file_path = os.path.join('/',os.path.join(*parts[:subject_id_position]),"image_paths.csv") 
        if image_class is None:
            print("No class is assigned to this button.")
                 
            return
        if image_class == "Path":
            print(f"Image {i} path is {image_path}")
            return
        try:
            if image_class == "Del":
                shutil.rmtree(image_path)
                print(f"Image {i} with path {image_path} has been deleted.")
                self.check_folder_isempty(image_path)
                self.check_folder_isempty(os.path.dirname(image_path))
                
            else:
#                if timepoints_present:
#                    j = 1
#                else:
#                    j = 0
                #### adjust so that when image is in ANP/CT it can move to HNC/CT
                
                if parts[-3] in labels_level1 and image_class in labels_level1:
                    class_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(image_path))),
                          image_class, os.path.basename(os.path.dirname(image_path)))
                elif parts[-3] in labels_level1 and not image_class in labels_level1:
                    class_dir = os.path.join(os.path.dirname(os.path.dirname(image_path)),image_class)
                        
                elif image_class in labels_level1:
                    if parts[-2] in classes:
                        dir_path, file_name = os.path.split(image_path)
                        parent_dir, current_dir = os.path.split(dir_path)
                        class_dir = os.path.join(parent_dir, image_class, current_dir)
                    else:
                        dir_path, file_name = os.path.split(image_path)
                        parent_dir, current_dir = os.path.split(dir_path)
                        class_dir = os.path.join(parent_dir, image_class)
                else:
                    if parts[-2] in labels_level1:
                        dir_path, file_name = os.path.split(image_path)
                        class_dir = os.path.join(dir_path, image_class)
                    else:
                        dir_path, file_name = os.path.split(image_path)
                        parent_dir, current_dir = os.path.split(dir_path)
                        class_dir = os.path.join(parent_dir, image_class)

                os.makedirs(class_dir, exist_ok=True)
                new_path = os.path.join(class_dir, os.path.basename(image_path))
                shutil.move(image_path, class_dir)
                print(f"Image {i} with path {image_path} has been moved to {class_dir}.")
                dicom_folders[i] = os.path.join(class_dir,image_path.split('/')[-1])
                # Remove empty directories
                self.check_folder_isempty(image_path)
                self.check_folder_isempty(os.path.dirname(image_path))
                self.update_csv(csv_file_path,new_path,image_class)

        except FileNotFoundError:
            print(f"The file '{image_path}' does not exist or is inaccessible.")
        except shutil.Error:
            print(f"The file '{image_path}' could not be moved to '{class_dir}', because the destination path already exists.")
            self.update_csv(csv_file_path,new_path,image_class)


    
def main(args):
    
    global classes
    # global dicom_folders
    # global subject_id_position
    # global labels_level1
    # labels_level1 = args.labels_level1
    # classes = labels_level1 + args.labels_level2 + ["Del", "Path","other"]
    # root_dir = args.root_dir
    # view = args.view
    # nii_convert = args.nii_convert
    # nr_images = args.nr_images
    labels_level1 = config['labels_level1']
    classes = labels_level1 + config['labels_level2'] + ["Del", "Path", "other"]
    root_dir = config['root_dir']
    view = config['view']
    nii_convert = config['nii_convert']
    nr_images = config['nr_images']    
    root_components = root_dir.split('/')
    
    subject_id_position = len(root_components) 
    root_dir = rearange_files(root_dir, subject_id_position)
    delete_small_directories(root_dir, minimun_slices = 9)
    
    dicom_folders = get_dicom_folders(root_dir)[:nr_images]

    ids = []
    for dicom_folder in dicom_folders:
        parts = dicom_folder.split('/')
        id = parts[subject_id_position]
        if id not in ids:
        # Print the ID and append it to the list
            ids.append(id)
        # Break the loop if three different IDs have been printed
        if len(ids) == 3:
            break
        # Print the patient IDs and ask the user to verify them
    print(f"Sample patient IDs found: {', '.join(ids)}")
    print("Please verify that these IDs match the expected patient IDs. If not, stop the application and check that the root direcorty contains the patient directories")
        
    dicom_folders = sorted(dicom_folders, key=lambda path: (path.split('/')[subject_id_position], 
                                                            path.split('/')[subject_id_position+1]))    
    patient_dict =  create_patient_dict(dicom_folders, id_pos = subject_id_position)
     # Load image and series description
     
    images, series_descriptions, orientation, modality, contrast, convert_failed = process_dicom_folders(dicom_folders, nii_convert, view)
    save_paths_to_csv(convert_failed)
    titles = []
    for dicom_folder, info in patient_dict.items():
       
        title = str(info[0])+"-"+str(info[1])+ "-" + \
        str(orientation[dicom_folders.index(dicom_folder)]) +"/ "+ \
        str(series_descriptions[dicom_folders.index(dicom_folder)]) + "_" + \
        str(contrast[dicom_folders.index(dicom_folder)])
        #print(dicom_folder)
        titles.append(title)

    
    app = QApplication(sys.argv)
    app.setStyleSheet("QLabel{font-size: 10pt;}")
    j = MainWindow(images, classes, titles,modality, dicom_folders, subject_id_position,labels_level1)
    print("HIIIIIIIIIIIIIIIIIIIIIIIII")
    sys.exit(app.exec_())


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('-r', '--root_dir', type=str, help='Path to the root directory of the DICOM files')
    # parser.add_argument('-l1', '--labels_level1', nargs='+', type=str, help='Labels used for level 2 classification')
    # parser.add_argument('-l2', '--labels_level2', nargs='+', type=str, help='Labels used for level 2 classification')
    # parser.add_argument('-v', '--view', type=str, help='View to display the images (coronal, axial or sagittal)')
    # parser.add_argument('-nii', '--nii_convert', type=bool, help='Whether to convert DICOM files to nii.gz or nrrd format')
    # parser.add_argument('-nr', '--nr_images', type=int, default=500, help='Number of images to plot. Default is set to 500')
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    #parser.add_argument('-p', '--subject_id_position', type=int, help='Position of the subject ID in the folder structure')

    #args = parser.parse_args()
    #main(args)
    main(config)