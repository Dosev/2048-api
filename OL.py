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
from keras.utils import multi_gpu_model


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

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i : i for i in range(1,CAND)}
map_table[0] = 0
vmap = np.vectorize(lambda x: map_table[x])

def one_hot(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)  # shape = (4,4,16)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret



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
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


Guide = namedtuple('Guide',('state','action'))

class Guides:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Guide(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def ready(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)

class ModelWrapper:
    def __init__(self,model,capacity):
        self.model = model
        self.memory = Guides(capacity)
        #self.writer = SummaryWriter()
        self.training_step = 0

    def predict(self, board):
        return model.predict(np.expand_dims(board,axis=0))

    def move(self, game):
        ohe_board = one_hot(vmap(game.board))
        suggest = board_to_move(game.board)
        direction = self.predict(ohe_board).argmax()
        game.move(direction)
        self.memory.push(ohe_board,suggest)

    def train(self, batch):
        if self.memory.ready(batch):
            guides = self.memory.sample(batch)
            X = []
            Y = []
            for guide in guides:
                X.append(guide.state)
                ohe_action = [0]*4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            loss,acc = self.model.train_on_batch(np.array(X),np.array(Y))
            #self.writer.add_scalar('loss',float(loss),self.training_step)
            #self.writer.add_scalar('acc',float(acc),self.training_step)
            if self.training_step % 1 == 0:
                with open('./naive_res_OL_new__true_final_6.txt', 'a') as f:
                    f.write('[%d] loss_val: %.05f'
                        % (self.training_step + 1,float(loss)))
                    f.write('[%d] acc_train: %.05f'
                        % (self.training_step + 1,float(acc)))
                    f.write('/n')
            self.training_step += 1
            if self.training_step % 100 == 0:
                model.save_weights("weights_finalOL_6_"+str(self.training_step)+".h5")

if __name__ == '__main__':
    model.load_weights('weights_finalOL_5_400.h5')
    tmp = ModelWrapper(parallel_model,32768)
    while True:
        game = Game(4, 2048)
        while not game.end:
            tmp.move(game)
        tmp.train(2048)     

