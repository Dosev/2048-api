import os
import numpy as np
from keras import losses
from keras.layers import Input, MaxPooling2D, UpSampling2D, Concatenate, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Model
from game2048.game import Game
import random
import math


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


#model.load_weights('weights_LOOP_1i6_itch_num200.h5')
#model.load_weights('weights_new_2400000.h5')
#model.load_weights('weights_2M_2.h5')
#model.load_weights('weights_new_LOOP_1i6_itch_num200.h5')
#model.load_weights('weights_newer_LOOP_120.h5')

#model.load_weights('weights_final_1100000.h5')
model.load_weights('weights_finalOL_5_400.h5')

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

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                #print("Iter: {}".format(n_iter))
                #print("======Direction: {}======".format(
                #    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyAgent(Agent):

    def __init__(self, game, display=None):
         super().__init__(game, display)
         
    def step(self):
        ohe_board = one_hot(vmap(self.game.board))
        direction = model.predict(np.expand_dims(ohe_board,axis=0)).argmax()
        return direction

'''class funAgent(Agent):
    def step(self):
        piece = [map_table[k] for k in self.game.board.astype(int).flatten().tolist()]
        print (one_hot(piece))
        print (piece)'''
