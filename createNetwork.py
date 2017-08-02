import pandas as pd
import sys, getopt
import numpy as np
import pprint
import os
from datetime import datetime
import glob
import pandas as pd
import emgimporter
import pickle
import time

filename = 'tobias_test_tools_v2'

# Stuck with np.roll as "scipy.ndimage.shift destroys data from edges (Trac #796)" -- gg
def augmentData(x,y,nb_roll,steps):
    assert len(x.shape) > 1
    x_cp = x
    y_cp = y
    for i in range(-nb_roll,nb_roll,steps):
        x_cp = np.vstack((x_cp,np.roll(x,i,axis=1)))
        y_cp = np.vstack((y_cp,y))

    return x_cp,y_cp


dic = { 1: 'Tasse aufnehmen',
        2: 'Tasse halten',
        3: 'Tasse abstellen',
        4: 'Tasse hoch&runter',
        8: 'Ruhe (Supination)',
        9: 'Ruhe (Pronation)'
      }
#its = ['cupv1','cupv2','kettlev1','data','loosev1','jascha','markus','Sirius', 'korbi','tobias','tobias2, tobias3']
its = ['cupv1','cupv2','kettlev1','data','jascha','korbi','loosev1','tobias','tobias2','tobias3','tobias4Tool']

trainStruct = { 'hidden_layer':      1,
                'neurons_per_layer': 20,
                'filter_layer':      8
                      }

data_arch = {   0: (7,8,9),
                1: (2,4),
             }

feed_dic = emgimporter.import_folder(its,dic,path=['emg_data','ipa_emg'])

x,y = emgimporter.prep_data(feed_dic,data_arch)

y_cut_train, y_cut_test, x_cut_train, x_cut_test = emgimporter.split_data(x,y,splitratio=0.05,shuffle=True)
x_cut_train= np.expand_dims(np.expand_dims(x_cut_train,axis=1),axis=-1)
x_cut_test= np.expand_dims(np.expand_dims(x_cut_test,axis=1),axis=-1)

x,y = augmentData(x_cut_train,y_cut_train,2,1)

# Force keras to use CPU ff
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.layers.convolutional import Conv1D
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected2D
from keras.layers import Conv2D, MaxPooling2D, LSTM, Conv1D, MaxPooling1D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.accuracy = []
    def on_epoch_end(self,epoch,logs={}):
        self.accuracy.append(logs.get('acc'))


saveList = list()

tb4 = datetime.now()

count = 0

fileList = [f for f in glob.glob(os.path.join(os.path.expanduser('~'),"network_tests",'**'),recursive=True) if not os.path.isdir(f)]
fileList.sort(key=lambda x: x[-15:])
saveList = pickle.load(open(fileList[-1],'rb'))

lk = max(saveList,key=lambda x:x[2])
ll = lk[0]

trainStruct['hidden_layer']       = ll[0]
trainStruct['filter_layer']       = ll[2]
trainStruct['neurons_per_layer']  = ll[1]

print("\n\n-------------------------------------------------------------------------")
print("Training network with Acc Score of ",lk[2], " and this structure: " )
print("neurons_per_layer: ", trainStruct['neurons_per_layer'])
print("hidden_layers:     ", trainStruct['hidden_layer'])
print("filter_layers:     ", trainStruct['filter_layer'])
print("-------------------------------------------------------------------------\n\n")


tb4 = datetime.now()
# Network architecture
output_classes    = 2
activation        = 'relu'
nb_filters        = trainStruct['filter_layer']
neurons_per_layer = trainStruct['neurons_per_layer']
hidden_layers     = trainStruct['hidden_layer']


model = Sequential()
model.add(Conv2D(filters=nb_filters,kernel_size=(1,8),input_shape=(1,8,1)))
model.add(Activation(activation))
model.add(Dropout(0.2))

model.add(Flatten())

for i in range(0,hidden_layers):
    model.add(Dense(neurons_per_layer))
    model.add(Activation(activation))
    model.add(Dropout(0.2))


if len(data_arch) < 0:
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], class_mode="binary" )
else:
    model.add(Dense(len(data_arch),activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], class_mode="sparse" )


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',patience=100,min_delta=0.0005)
history = AccuracyHistory()

model.fit(x,y,epochs = 150, batch_size = 3000, validation_split=0.1, shuffle=True, callbacks = [early_stopping,history], verbose=1)
scores = model.evaluate(x_cut_test, y_cut_test, batch_size=1000, verbose=1)
tbn = datetime.now()
print("\n------------------------")
print("timestamp: ", tbn , " | Time diff: ", tbn - tb4)
tb4 = tbn
print("Architecture: ")
print("Hidden Layers: ", hidden_layers)
print("Neurons:       ", neurons_per_layer)
print("ConvFilters:   ", nb_filters)
print("\nScore: ", scores[1], " | Loss: ", scores[0])
print("\n")

fileName = "network_" + str(ll[0]) + "_" +  str(ll[1]) + "_" + str(ll[2]) + "_"+ str(int(10000*scores[1])) +".h5"

model.save(os.path.join("/home/myo/models/",fileName))
