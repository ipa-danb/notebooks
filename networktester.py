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
import importlib

filename = 'default'
configFile = None

def augmentData(x,y,nb_roll,steps):
    assert len(x.shape) > 1
    x_cp = x
    y_cp = y
    for i in range(-nb_roll,nb_roll,steps):
        x_cp = np.vstack((x_cp,np.roll(x,i,axis=1)))
        y_cp = np.vstack((y_cp,y))

    return x_cp,y_cp

def padStuff(matrix,expsize=(2,2),axis=0,mode='wrap'):
    dim = len(matrix.shape)
    tup1 = [(0,0) for _ in range(0,dim)]
    tup1[axis] = expsize
    return np.pad(matrix,tup1,mode)

class default_config:
    filename = 'default'
    loadstuff = False
    dic = { 1: 'Tasse aufnehmen',
            2: 'Tasse halten',
            3: 'Tasse abstellen',
            4: 'Tasse hoch&runter',
            8: 'Ruhe (Supination)',
            9: 'Ruhe (Pronation)'
          }
    its = ['cupv1','cupv2','kettlev1','data','loosev1','jascha','markus','Sirius', 'korbi','tobias_1','tobias_2']
    explorationStruct = { 'hidden_layer':      [1,2,1],
                          'neurons_per_layer': [100,200,25],
                          'filter_layer':      [75,150,25]
                          }
    model_params = { 'model':'CNN' # or CNN

                    }
    data_arch = {   0: (7,8,9),
                    1: (2,4),
                 }
    convFilterDict = { "conv":  {
                                "filters":       30,
                                "kernel_size":   (1,6),
                                "input_shape":   (1,10,1)
                                },
                       "maxpool":{"pool_size":(1,5)}
                     }
    path = ['emg_data','ipa_emg']
    output_classes = 2
    activation = 'relu'

opts, args = getopt.getopt(sys.argv[1:],"hc:")
for o,a in opts:
    if o == "-h":
        print("Use -l to load data from latest file in default folder (~/network_tests)")
        sys.exit()
    elif o == "-c":
        print("Using config file: ",a)
        configFile = a


if configFile == None:
    config = default_config
else:
    try:
        config = importlib.import_module(configFile)

    except:
        k = input( str(configFile) +" doesnt exist, switching to default[y] or abort[n]? [y/n]" )
        if k == "y":
            config = default_config
        else:
            print("Exiting...")
            sys.exit(2)

print("Examinating Neural Networks. \nSaving with ", config.filename)

# load data
feed_dic = emgimporter.import_folder(config.its,config.dic,path=config.path)

# prep data
x,y = emgimporter.prep_data(feed_dic,config.data_arch)
if kfold == None:
    y_cut_train, y_cut_test, x_cut_train, x_cut_test = emgimporter.split_data(x,y,splitratio=0.05,shuffle=True)
    x_cut_train= np.expand_dims(np.expand_dims(x_cut_train,axis=1),axis=-1)
    x_cut_test= np.expand_dims(np.expand_dims(x_cut_test,axis=1),axis=-1)
else:
    kfold_list = emgimporter.split_data(x,y,shuffle=True,option='kfold',kfold= 6)

    for i,element in enumerate(kfold_list[0]):
        x_kfold_prepped.append( np.expand_dims(np.expand_dims(element,axis=1),axis=-1) )




# augment data
x,y = augmentData(x_cut_train,y_cut_train,2,1)
x = padStuff(x,expsize=(1,1),axis=2,mode='wrap')
x_cut_test = padStuff(x_cut_test,expsize=(1,1),axis=2,mode='wrap')


# Load everything keras
from keras.layers.convolutional import Conv1D
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected2D
from keras.layers import Conv2D, MaxPooling2D, LSTM, Conv1D, MaxPooling1D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# class to log history
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.accuracy = []
    def on_epoch_end(self,epoch,logs={}):
        self.accuracy.append(logs.get('acc'))


saveList = list()
tb4 = datetime.now()
count = 0

# load progress from files if necessary
if config.loadstuff:
    fileList = [f for f in glob.glob(os.path.join(os.path.expanduser('~'),"network_tests",'**'),recursive=True) if not os.path.isdir(f)]
    fileList.sort(key=lambda x: x[-15:])
    saveList = pickle.load(open(fileList[-1],'rb'))
    explorationStruct['hidden_layer'][0]      = saveList[-1][0][0]
    explorationStruct['filter_layer'][0]      = saveList[-1][0][2]
    explorationStruct['neurons_per_layer'][0]  = saveList[-1][0][1]

print("\n\n-------------------------------------------------------------------------")
print("starting at ")
print("neurons_per_layer: ", config.explorationStruct['neurons_per_layer'][0])
print("hidden_layers:     ", config.explorationStruct['hidden_layer'][0])
print("filter_layers:     ", config.explorationStruct['filter_layer'][0])
print("-------------------------------------------------------------------------\n\n")
print("\n\n-------------------------------------------------------------------------")
print("Amount of loaded data: ")
print("Samples for training: ", x_cut_train.shape[0])
print("Samples for testing:  ", x_cut_test.shape[0])
print("-------------------------------------------------------------------------\n\n")


# check if data is set up correctly
assert(x.shape[1:] == x_cut_test.shape[1:])

# calculate input_shape
config.convFilterDict["conv"]["input_shape"] = x.shape[1:]
config.convFilterDict["maxpool"]["pool_size"] = (1, config.convFilterDict["conv"]["input_shape"][1] - config.convFilterDict["conv"]["kernel_size"][1] + 1)

print("\n\n-------------------------------------------------------------------------")
print("ConvFilterDict: ")
print(config.convFilterDict)
print("-------------------------------------------------------------------------\n\n")

tst = datetime.now()
for hidden_layers in range(*config.explorationStruct['hidden_layer']):
    for nb_filters in range(*config.explorationStruct['filter_layer']):
        for neurons_per_layer in range(*config.explorationStruct['neurons_per_layer']):
            # Network architecture
            output_classes    = config.output_classes
            activation        = config.activation
            config.convFilterDict["conv"]["filters"] = nb_filters

            model = Sequential()
            if config.model_params['model'] == 'CNN':
                model.add(Conv2D(**config.convFilterDict["conv"]))
                model.add(MaxPooling2D(**config.convFilterDict["maxpool"]))
            elif config.model_params['model'] == 'MLP':
                model.add(Dense(neurons_per_layer,input_shape=config.convFilterDict["conv"]["input_shape"]))
            model.add(Activation(config.activation))
            model.add(Dropout(0.2))

            model.add(Flatten())

            for i in range(0,hidden_layers):
                model.add(Dense(neurons_per_layer))
                model.add(Activation(config.activation))
                model.add(Dropout(0.2))


            if len(config.data_arch) < 0:
                model.add(Dense(1,activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], class_mode="binary" )
            else:
                model.add(Dense(len(config.data_arch),activation='softmax'))
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], class_mode="sparse" )


            from keras.callbacks import EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss',patience=50,min_delta=0.001)
            history = AccuracyHistory()

            # learn model and plot show progress
            if kfold == None:
                model.fit(x,y,epochs = 1000, batch_size = 3000, validation_split=0.1, shuffle=True, callbacks = [early_stopping,history], verbose=0)
                scores = model.evaluate(x_cut_test, y_cut_test, batch_size=1000, verbose=0)
            else:

            tbn = datetime.now()

            # Print pretty stuff
            print("\n------------------------")
            print("timestamp: ", tbn , " | Time diff: ", tbn - tb4)
            tb4 = tbn
            print("Architecture: ")
            print("Hidden Layers: ", hidden_layers)
            print("Neurons:       ", neurons_per_layer)
            print("ConvFilters:   ", nb_filters)
            print("\nScore: ", scores[1], " | Loss: ", scores[0])
            print("\n")

            # save progress to list in memory
            tmp = ( (hidden_layers, neurons_per_layer, nb_filters) ,scores[0], scores[1], history.accuracy )

            saveList.append( tmp )
            count += 1

            # save progress to file on hdd (for resuming)
            if (tbn-tst).total_seconds() > 15*60 or count > 100:
                tst = datetime.now()
                timestr = time.strftime("%Y%m%d-%H%M%S")
                pickle.dump(saveList,  open('/home/myo/network_tests/' + config.filename + timestr, 'wb'))
                count = 0


timestr = time.strftime("%Y%m%d-%H%M%S")

# save last file, if finished
pickle.dump(saveList,  open('/home/myo/network_tests/' + config.filename + timestr + "_final", 'wb'))
