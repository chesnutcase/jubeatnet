import sys
from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

import cv2 
import pandas as pd
from jubeatnet.parsers import CosmosMemoParser
from pathlib import Path
import copy
import numpy as np

class MainWindow(QMainWindow):
	# Pages Index
    HOMEPAGE = 1
    ANNOTATEPAGE = 0

    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi('JubeatAnnotator.ui',self)
        self.startButton.clicked.connect(self.start)
        self.nextButton.clicked.connect(self.next)
        self.backButton.clicked.connect(self.back)
        self.completeButton.clicked.connect(self.complete)
        self.saveButton.clicked.connect(self.save)
        self.MEMODIR = "resources/"
        self.scene = QGraphicsScene()
        self.name = None
        self.firsthit = 0
        self.beatinfo = None
        self.frames = []
        self.count = 0 
        self.label = []
        self.grid = [[self.grid_1, self.grid_2, self.grid_3, self.grid_4],
                     [self.grid_5, self.grid_6, self.grid_7, self.grid_8],
                     [self.grid_9, self.grid_10, self.grid_11, self.grid_12],
                     [self.grid_13, self.grid_14, self.grid_15, self.grid_16]]
        self.template = [[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]]

        self.stackedWidget.setCurrentIndex(self.HOMEPAGE)
        self.stackedWidget.setMouseTracking(True)

    def clearparams(self): 
        self.name = None
        self.firsthit = 0 
        self.beatinfo = None 
        self.frames = [] 
        self.count = 0 
        self.label = []
        self.scene = QGraphicsScene()
        # self.scene.addPixmap(None)
        self.imgView.setScene(self.scene)

    # Get frames for hand detection
    def get_hand_frames(self,beatinfo,firsthit,fps,vidfile):
        cap = cv2.VideoCapture(vidfile)  
        frames = []
        
        firsthit_frame = int(firsthit*fps)
        accum = copy.deepcopy(firsthit_frame)
        reference = []
        for pointer in range(len(beatinfo)):
            reference.append(accum)
            if np.isinf(beatinfo[pointer][0]):
                break
            accum+=int(beatinfo[pointer][0]*fps)
        
        ref_pointer = 0
        count = 0 
        while True:
            _, frame = cap.read()
            # Record first hit 
            if count == reference[ref_pointer]: 
                # add to list
                frames.append(frame)
                if ref_pointer == len(reference)-1:
                    break
                ref_pointer += 1
            count += 1    
            if frame is None:
                break
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        return frames


    def read_memo(self,file): 
        memo = Path(file)
        parser = CosmosMemoParser()
        chart = parser.parse(memo)
        beat = chart.to_numpy_array()
        return beat

    def get_fps(self,file):
        cap = cv2.VideoCapture(file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps

    def load_image(self):
        frame = self.frames[self.count]
        image = QImage(frame, frame.shape[1], frame.shape[0],
                  frame.strides[0], QImage.Format_RGB888)
        self.scene = QGraphicsScene()
        self.scene.addPixmap(QPixmap.fromImage(image))
        self.imgView.setScene(self.scene)


    def load_grid(self): 
        self.cleargrid()
        # if current count > len-1, dont do shit 
        # else set as whatever is there
        if self.count <= len(self.label)-1:
            # get grid 
            vals = self.label[self.count]
            # set the grid 
            for i in range(4):
                for j in range(4): 
                    self.grid[i][j].setText(str(vals[i][j]))

    def cleargrid(self): 
        for i in self.grid:
            for j in i: 
                j.setText("0")

    def load_beat(self):
        self.beatLabel.setText(np.array_str(self.beatinfo[self.count][1]))
        
    def savelabels(self):
        print("saving")
        print(len(self.label))
        np.save("out/{}".format(self.name),np.array(self.label))

    def handle_grid(self):
        grid = copy.deepcopy(self.template)
        for i in range(4): 
            for j in range(4): 
                grid[i][j] = int(self.grid[i][j].text())
        if self.count <= len(self.label)-1:
            self.label[self.count] = np.array(grid)
        else:
            self.label.append(np.array(grid))

    #################
    # Button Events #
    #################
    def start(self):
        # Get vid info 
        vidfile,filter = QFileDialog.getOpenFileName(self,'Open File',"","Video Files(*.mkv     *.mp4)")
        self.name = vidfile.split("/")[-1].split(".")[0]
        df = pd.read_excel("resources/info.xlsx")
        df = df[["song_name","first_hit_time"]]
        self.firsthit = df[df["song_name"] == self.name]["first_hit_time"].values[0] 
        self.beatinfo= self.read_memo(self.MEMODIR + "{}.txt".format(self.name.lower().replace(" ","_")))
        fps = self.get_fps(vidfile)
        self.frames = self.get_hand_frames(self.beatinfo,self.firsthit,fps,vidfile)
        #self.frames = self.frames[:5]
        # Init labels if already exist 
        try: 
            self.label = list(np.load("out/"+self.name+".npy"))
            self.count = len(self.label)
        except:
            print("Oops this file aint exist")
        # Load stuff 
        self.load_image() 
        self.load_grid()
        self.load_beat()
        # Change page
        self.stackedWidget.setCurrentIndex(self.ANNOTATEPAGE)
        self.backButton.setEnabled(False)
        self.completeButton.setEnabled(False)
        self.nextButton.setEnabled(True)


    def next(self):
        # Add to the label list
        self.handle_grid()
        # Go to next item 
        self.count += 1
        self.load_image() 
        self.load_grid()
        self.load_beat()
        print(self.label)
        # Check button
        if self.count == len(self.frames)-1:
            self.nextButton.setEnabled(False)
            self.completeButton.setEnabled(True)
        self.backButton.setEnabled(True)

        

    def back(self):
        # Add to the label list
        self.handle_grid() 
        # Go to prev item 
        self.count -= 1
        self.load_image() 
        self.load_grid() 
        self.load_beat() 
        print(self.label)
        # Check button 
        if self.count != len(self.frames)-1: 
            self.nextButton.setEnabled(True)
            self.completeButton.setEnabled(False)
        if self.count == 0:
            self.backButton.setEnabled(False)

    def complete(self):
        self.handle_grid() 
        # Clear everything 
        self.cleargrid() 
        self.savelabels()
        self.clearparams()
        # Set buttons back 
        self.backButton.setEnabled(False)
        self.completeButton.setEnabled(False)
        self.nextButton.setEnabled(True)
        # Change page
        self.stackedWidget.setCurrentIndex(self.HOMEPAGE)

    def save(self):
        self.handle_grid()
        self.savelabels()


#################
#    Main()     #
#################
if __name__ == '__main__':
    app=QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())