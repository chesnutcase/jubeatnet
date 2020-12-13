import sys
import os 
from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import time
from tqdm import tqdm

import cv2 
import pandas as pd
from pathlib import Path
import copy
import numpy as np

class ProcessWorker(QObject):
    imageChanged = pyqtSignal(QImage)

    def __init__(self,frames):
        super(ProcessWorker, self).__init__()
        self.frames = frames

    def doWork(self):
        for i in range(len(self.frames)):
            frame = self.frames[i]
            image = QImage(frame, frame.shape[1], frame.shape[0],
                  frame.strides[0], QImage.Format_RGB888)
            self.imageChanged.emit(image)
            QThread.msleep(1000)

class MainWindow(QMainWindow):
	# Pages Index
    HOMEPAGE = 1
    VISUALIZEPAGE = 0

    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi('jubeatvisualize.ui',self)
        self.startButton.clicked.connect(self.start)
        self.runButton.clicked.connect(self.run)
        self.openButton.clicked.connect(self.open)
        self.modelBox.currentIndexChanged.connect(self.on_modelbox_changed)
        self.modelBox.setCurrentIndex(0)
        self.model = self.modelBox.currentIndex
        self.TRAIN_DIR_DATA = "./train/data/"
        self.TRAIN_DIR_LABEL = "./train/label/"
        self.CNNMODELFILEPATH = "model.h5"
        self.scene = QGraphicsScene()
        self.data = None
        self.frames = []
        self.blank = np.zeros((400,400,3), np.uint8)
        self.blank[self.blank == 0] = 255

        self.stackedWidget.setCurrentIndex(self.HOMEPAGE)
        self.stackedWidget.setMouseTracking(True)


    #################
    # Button Events #
    #################
    def start(self):
        # Change page
        self.stackedWidget.setCurrentIndex(self.VISUALIZEPAGE)

    def run(self):
        if self.model == 0:
            output = self.CNN(self.data)
        else: 
            output = self.HMM(self.data)
        self.output_to_frames(output)
        self.run_video()


    def open(self):
        npfile,filter = QFileDialog.getOpenFileName(self,'Open File',"","Numpy Files(*.npy)")
        self.data = np.load(npfile)
            
    def on_modelbox_changed(self):
        self.model = self.modelBox.currentIndex

    def CNN(self,input_beat):
        input_height = 4
        input_width = 4
        n_classes = 11
        img_size = (4, 4)
        num_classes = 11
        batch_size = 8

        img_input = Input(shape=(input_height,input_width,1))

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

        up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

        up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

        out = Conv2D( n_classes, (1, 1) , padding='same')(conv5)

        model = keras.Model(img_input, out)
        model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy')

        model.load_weights(self.CNNMODELFILEPATH)

        test_gen = JubeatDataset(batch_size,img_size,input_beat,input_beat)
        test_preds = model.predict(test_gen)
        output = []
        for array in test_preds: 
            output.append(np.argmax(array, axis=-1))
        output = np.array(output)
        return output

    def HMM(self,input_beat):
        train_data,train_label = self.get_data(self.TRAIN_DIR_DATA)
        e_array, y_e, x_e, t_array, y_t, x_t = self.train(train_data,train_label)
        test = np.array(self.deconstruct_grid(input_beat))
        output = self.predict(test,e_array, y_e, x_e, t_array, y_t, x_t)
        output = np.array(output)
        return output

    def output_to_frames(self,output):
        for i in range(len(output)):
            self.frames.append(self.array_to_img(output[i]))

    def array_to_img(self,array):
        image = np.zeros((400,400,3), np.uint8)
        image[image == 0] = 255
        image = cv2.rectangle(image,(0,0),(399,399),(0,0,0),1)
        for i in range(4): 
            for j in range(4):
                image = cv2.rectangle(image,(i*100,j*100),((i+1)*100,(j+1)*100),(0,0,0),1)

        pos = np.array(np.where(array>0)).transpose()
        for i in pos: 
            image = cv2.rectangle(image,(i[0]*100,i[1]*100),((i[0]+1)*100,(i[1]+1)*100),(255,0,0),-1)
            image = cv2.putText(image, str(array[i[0]][i[1]]), (i[0]*100+40,i[1]*100+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
        return image


    def run_video(self):
        self.workerThread = QThread()
        self.worker = ProcessWorker(self.frames)
        self.worker.moveToThread(self.workerThread)
        self.workerThread.finished.connect(self.worker.deleteLater)
        self.workerThread.started.connect(self.worker.doWork)
        self.worker.imageChanged.connect(self.setImage)
        self.workerThread.start()

    def setImage(self,image):
            self.scene = QGraphicsScene()
            self.scene.addPixmap(QPixmap.fromImage(image))
            self.imgView.setScene(self.scene)


    ###################
    #       HMM       #
    ###################

    def get_data(self,data_dir):
        data_files = os.listdir(data_dir)
        data = []
        label = []
        for dfile in data_files:
            data.append(np.load(self.TRAIN_DIR_DATA + dfile))
            label.append(np.load(self.TRAIN_DIR_LABEL + dfile.split(".")[0] + " label.npy"))
        return data,label

    def pre_process(self,data,label):
        files_pos = []
        files_fingers = []
        for l in label: 
            grid_pos = []
            beat_fingers = []
            for j in range(len(l)):
                pos = np.array(np.where(l[j]>0)).transpose()
                grid_pos.append(pos)
                beat_fingers.append([l[j][i[0]][i[1]] for i in pos])
            files_pos.append(grid_pos)
            files_fingers.append(beat_fingers)
        return files_pos, files_fingers

    def get_eprobs(self,val,finger,fingers):
        return val/fingers.count(finger)


    def get_emission(self,files_pos,files_fingers):
        pos_int = None
        fingers_int = None
        for i in range(len(files_pos)): 
            if i == 0: 
                pos_int = files_pos[i]
                fingers_int = files_fingers[i]
            else: 
                pos_int += files_pos[i] 
                fingers_int += files_fingers[i]

        pos = []
        list(map(pos.extend, pos_int))
        fingers = [] 
        list(map(fingers.extend, fingers_int))
        pos = [tuple(i) for i in pos]

        emission_df = pd.DataFrame()
        emission_df["pos"] = pos
        emission_df["fingers"] = fingers

        df = emission_df.groupby(['pos', 'fingers']).size().reset_index()
        df.columns = ["pos","fingers","count"] 


        df["e"] = df.apply(lambda x: self.get_eprobs(x['count'],x['fingers'],fingers),axis=1)
        df = df.drop(columns = ["count"])
        
        return df

    def get_tprobs(self,val,finger,new_prev_finger):
        return val/np.count_nonzero(new_prev_finger.flatten() == 1)

    def get_transition(self,files_pos,files_fingers):
        post_file_finger = []
        transmission_df = pd.DataFrame()

        prev_finger = []
        curr_finger = []
        for i in range(len(files_pos)): 
            prev_finger_temp = files_fingers[i][0:len(files_fingers[i])-1]
            curr_finger_temp = files_fingers[i][1:len(files_fingers[i])]
            prev_finger += prev_finger_temp
            curr_finger += curr_finger_temp

        prev_finger = [tuple(sorted(i)) for i in prev_finger]
        curr_finger = [tuple(sorted(i)) for i in curr_finger]

        new_prev_finger = []
        new_curr_finger = []

        for tuplei,tuplej in zip(prev_finger,curr_finger): 
            for val1 in tuplei: 
                for val2 in tuplej:
                    new_prev_finger.append(val1)
                    new_curr_finger.append(val2)

        new_prev_finger = np.array(new_prev_finger)
        new_curr_finger = np.array(new_curr_finger)


        transmission_df["prev_finger"] = new_prev_finger
        transmission_df["curr_finger"] = new_curr_finger
        transmission_df
        df1 = transmission_df.groupby(['prev_finger', 'curr_finger']).size().reset_index()
        df1 = df1[df1["prev_finger"] != () ]
        df1 = df1[df1["curr_finger"] != () ]
        # df1

        df1.columns = ["prev_finger", "curr_finger", "count"]
        df1
        df1["t"] = df1.apply(lambda x: self.get_tprobs(x['count'],x['prev_finger'],new_prev_finger),axis=1)
        df1 = df1.drop(columns = ["count"])
        return df1

    def emission_to_table(self,df):
        row_e = df["pos"].values
        column_e = df["fingers"].values
        y_e = list(np.unique(row_e))
        x_e = list(np.unique(column_e))
        # Init empty array 
        e_array = np.zeros((len(y_e),len(x_e)))

        # Fill up 
        for i in range(len(df)):
            row_idx = y_e.index(df.iloc[i]["pos"])
            column_idx = x_e.index(df.iloc[i]["fingers"])
            e_array[row_idx][column_idx] = df.iloc[i]["e"]

        return e_array, y_e, x_e

    def transition_to_table(self,df1): 
        row_t = df1["prev_finger"].values
        column_t = df1["curr_finger"].values
        y_t = list(np.unique(row_t))
        x_t = list(np.unique(column_t))

        # Init empty array
        t_array = np.zeros((len(y_t),len(x_t)))

        # Fill up 
        for i in range(len(df1)):
            row_idx = y_t.index(df1.iloc[i]["prev_finger"])
            column_idx = x_t.index(df1.iloc[i]["curr_finger"])
            t_array[row_idx][column_idx] = df1.iloc[i]["t"]
        return t_array,y_t,x_t

    def reconstruct_grid(self,grids,fingers):
        finger_grid = np.zeros((4,4))
        for i,j in zip(grids,fingers):
            if j == None:
                finger_grid[i[0]][i[1]] = 0
            else: 
                finger_grid[i[0]][i[1]] = j
        return finger_grid

    def train(self,data,label):
        files_pos,files_fingers = self.pre_process(data,label)
        e_df = self.get_emission(files_pos,files_fingers)
        t_df = self.get_transition(files_pos,files_fingers)
        e_array, y_e, x_e = self.emission_to_table(e_df)
        t_array,y_t,x_t = self.transition_to_table(t_df)
        return e_array, y_e, x_e, t_array, y_t, x_t
        
    def predict_one(self,grid_pos_list, prev_finger, e_array, y_e, x_e, t_array, y_t, x_t):
        chosen_fingers = []
        y_e = np.array(y_e)
        x_e = np.array(x_e)
        y_t = np.array(y_t)
        x_t = np.array(x_t)
        
        for grid_pos in grid_pos_list: 
            chosen_finger = None
            scores = [1 for i in range(10)]

            if prev_finger != None:
                for finger in prev_finger: 
                    for i in range(10):
                        try: 
                            y = np.where(np.array(y_t) == finger)[0][0]
                            x = np.where(np.array(x_t) == i+1)[0][0]
                            t = t_array[y][x]
                            scores[i] *= t
                        except: 
                            scores[i] *= 0.001
            for i in range(10):
                try: 
                    y = np.where(np.array(y_e) == grid_pos)[0][0]
                    x = np.where(np.array(x_t) == i+1)[0][0]
                    e = e_array[y][x]
                    scores[i] *= e
                except: 
                    scores[i] *= 0.001
            scores_copy = sorted(copy.deepcopy(scores))
            idxes = [scores.index(i)+1 for i in scores_copy]
            for chosen in idxes:
                if chosen not in chosen_fingers: 
                    chosen_finger = copy.deepcopy(chosen) 
            chosen_fingers.append(chosen_finger)
        return chosen_fingers



    def predict(self,grids_sequence,e_array, y_e, x_e, t_array, y_t, x_t): 
        prev_fingers = None
        finger_sequence = []
        
        for beat in tqdm(grids_sequence):
            curr_fingers = self.predict_one(beat,prev_fingers,e_array, y_e, x_e, t_array, y_t, x_t)
            prev_fingers = copy.deepcopy(curr_fingers)
            finger_sequence.append(curr_fingers)
        
        finger_array = []
        for pos,finger in zip(grids_sequence, finger_sequence):
            finger_array.append(self.reconstruct_grid(pos,finger))
        
        return finger_array

    def deconstruct_grid(self,grid_arrays):
        grid_pos = []
        for l in range(len(grid_arrays)): 
            pos = np.array(np.where(grid_arrays[l]>0)).transpose()
            grid_pos.append(pos)
        return grid_pos



#################
#    Main()     #
#################
if __name__ == '__main__':
    app=QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
