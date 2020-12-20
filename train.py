import csv
import os
import numpy as np
from keras import losses
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from game2048.expectimax import board_to_move
from collections import namedtuple
from game2048.game import Game
import random
from keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.convolutional import Conv2D
import math
import csv
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os.path
from keras import optimizers
import keras

#train_file = '/data2M.csv'
BATCH_SIZE = 1024


        

'''def one_hot(boards):
    a = []
    b = []
    c = []
    for i in range(4):
        a.append(b)
    for j in range(4):
        c.append(a)

    for i in range(4):
        for j in range(4):
            if boards[i][j] != 0:
                boards[i][j] = math.log(boards[i][j],2)
            tmp = c
            zero_list = [0] * 16
            zero_list[int(boards[i][j])] = 1
            tmp[i][j] = zero_list
    board = tmp
    
    return board'''

'''def read_csv(csv_file):
    truth_arr = []
    with open(csv_file) as csvfile:
	    csv_reader = csv.reader(csvfile)
	    for row in csv_reader:  
	            truth_arr.append(row)
    truth_arr = np.array(truth_arr)
    label = truth_arr[:,16]
    Y = []
    print (label.shape)
    for i in range(len(label)):
        ohe_action = [0]*4
        ohe_action[int(label[i][0])] = 1
        Y.append(ohe_action)
    Y = np.asarray(Y)
    print (Y.shape)
    boards = truth_arr[:,0:16].reshape(-1, 4)
    res = []
    for num in range(int(len(boards)/4)):
        a = []
        for i in range(4):
            b = []
            a.append(b)
        tmp = a
        for i in range(4):
            for j in range(4):
                zero_list = [0] * 16
                zero_list[int(boards[i][j])] = 1
                tmp[i].append(zero_list)
        res.append(tmp)
    res = np.asarray(res)
    print (res.shape)
    return res, Y'''

OUT_SHAPE = (4,4)
CAND = 16

def grid_one(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)  # shape = (4,4,16)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret

def got_x_y(train_file):
    data_raw = []
    with open(train_file) as f:
        for line in f:
            piece = eval(line)
            data_raw.append(piece)
    data = np.array(data_raw)
    x = np.array([ grid_one(piece[:-1].reshape(4,4)) for piece in data ])
    y = keras.utils.to_categorical(data[:,-1], 4)
    return x, y


inputs = Input(shape = (4,4,16))

conv = inputs
FILTERS = 128
conv41 = Conv2D(filters=FILTERS,kernel_size=(4,1),kernel_initializer='he_uniform')(conv)
conv14 = Conv2D(filters=FILTERS,kernel_size=(1,4),kernel_initializer='he_uniform')(conv)
conv22 = Conv2D(filters=FILTERS,kernel_size=(2,2),kernel_initializer='he_uniform')(conv)
conv33 = Conv2D(filters=FILTERS,kernel_size=(3,3),kernel_initializer='he_uniform')(conv)
conv44 = Conv2D(filters=FILTERS,kernel_size=(4,4),kernel_initializer='he_uniform')(conv)

hidden = Concatenate()([Flatten()(conv41),Flatten()(conv14),Flatten()(conv22),Flatten()(conv33),Flatten()(conv44)])
x = BatchNormalization()(hidden)
x = Activation('relu')(x)

for width in [512,128]:
    x = Dense(width,kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

outputs = Dense(4,activation='softmax')(x)
model = Model(inputs,outputs)
model.summary()
#sgd = optimizers.SGD(lr = 0.005, decay = 1e-5)
#model.compile(optimizer=sgd,loss = 'categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

#print ("board",board_arr[0], "label",label_arr[0])
#print (board_arr.shape, label_arr.shape)
#itch_num = 0
#checkpoint = ModelCheckpoint(filepath='./',save_best_only='False',period=10)
#tensorboard = TensorBoard(log_dir='./')
#callback_lists=[tensorboard,checkpoint]
#while True:

#model.load_weights('weights_2M_1.h5')
LOOP = 0
while True:
    LOOP += 1
    for i in range(1,16):
        train_file = "./train/train1M_" + str(i) + ".csv"
        board_arr, label_arr = got_x_y(train_file)
        itch_num = 0
        model.fit(board_arr, label_arr, batch_size = BATCH_SIZE,epochs = 1, verbose = 2)
    if LOOP % 10 == 0:
        #model.save_weights("weights_newer_LOOP_" + str(LOOP) + "i"+ str(i) + "_itch_num" + str(itch_num) + ".h5")
        model.save_weights("weights_newer_LOOP_" + str(LOOP) + ".h5")

'''iter_num = 0
train_file = "all_with_15M.csv"
board_arr, label_arr = got_x_y(train_file)
while True:
    model.fit(board_arr, label_arr, batch_size = BATCH_SIZE,epochs = 10, verbose = 2)
    model.save_weights("weights_15M_" + str(iter_num) + ".h5")
    iter_num += 1'''
    
model.save_weights("weights_.h5")
    
    


