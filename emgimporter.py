import os
from datetime import datetime
import glob
import pandas as pd
import numpy as np
import math

def sfun(string):
    """
    Extracts timestring from EMG data (MYO Armband)
    """
    _,date = string.split('-',1)
    return datetime.strptime(date,"%Y%m%d-%H%M%S")

def lexp(l):
    """
    Expands a list of paths with emg data into a list of tuples with name,date and filename of each
    """
    l2 = list()
    for f in l:
        name,date = f.split('-',1)
        l2.append((name,datetime.strptime(date,"%Y%m%d-%H%M%S"),f))
    return l2


def import_data(name=None,path=[]):
    """
    Imports EMG data CSVs recursively.
    Given path is respective to home user path
    """
    fList = [(os.path.basename(f.split('-',1)[0]), datetime.strptime(f.split('-',1)[1],"%Y%m%d-%H%M%S"), f)
              for f in glob.glob(os.path.join(os.path.expanduser('~'),*path,'**'),recursive=True)
              if not f.startswith('.') and not os.path.isdir(f)]
    try:
        return [pd.read_csv(f[2], skiprows=1, header=None, delim_whitespace=True) for f in fList  if name == None or f[0] == name]
    except:
        print("well... ", name)
        pass

def prep_data(feed_dic,data_arch):
    x = list()
    y = list()
    for f in feed_dic:
        for k in feed_dic[f]:
            if k[1]:
                a = np.concatenate([np.array(i) for i in k[1] ], axis= 0 )
                b = np.resize(np.array(k[0]), (a.shape[0],1))
                x.append(a)
                y.append(b)

    y = np.concatenate(y)
    x = np.concatenate(x)

    # extract all categories into tuple
    l = list()
    for f in data_arch:
        l.extend([*data_arch[f]])
    l = tuple(l)

    # cut samples that belong to all categories from all samples
    x_cut = x[ np.in1d(y[:,0], l) ]
    y_cut = y[ np.in1d(y[:,0], l) ]

    # replace gesture IDs with category IDs
    for f in data_arch:
        y_cut[ np.in1d(y_cut[:,0], data_arch[f] ) ] = f

    # encode categories in one-hots format
    from keras.utils.np_utils import to_categorical
    y_cut_binary = to_categorical(y_cut)

    return x_cut,y_cut_binary

def import_folder(its, dic, path = [] ):
    """
    Import folders recursively with filenames in its and gestures to import in dic
    """
    feed_dic = dict()
    for n in its:
        l2 = list()
        for i in dic:
             l2.append((i,import_data(n+'_'+str(i),path=path)))
        feed_dic[n] = l2
    return feed_dic

def unison_shuffled_copies(a, b):
    """
    Shuffle two arrays randomly
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def split_data(x_cut,y_cut_binary, option = 'split',splitratio = 0.1, shuffle = True, kfold = 4):

    # shuffle data by default
    if shuffle:
        x_cut, y_cut_binary = unison_shuffled_copies(x_cut,y_cut_binary)

    # split-off test data
    if option == 'split':
        split_size = int(x_cut.shape[0]*splitratio)
        y_cut_train, y_cut_test = y_cut_binary[split_size:], y_cut_binary[:split_size]
        x_cut_train, x_cut_test = x_cut[split_size:], x_cut[:split_size]
        return y_cut_train, y_cut_test, x_cut_train, x_cut_test
    elif option == 'kfold':
        x_kfold_retVal = list()
        y_kfold_retVal = list()
        split_size = int(math.floor(x_cut.shape[0]/kfold))
        for i in range(0,kfold-1):
            x_kfold_retVal.append(x_cut[i*split_size:(i+1)*split_size])
            y_kfold_retVal.append(y_cut_binary[i*split_size:(i+1)*split_size])
        x_kfold.retVal.append(x_cut[(kfold-1)*split_size:])
        y_kfold.retVal.append(y_cut_binary[(kfold-1)*split_size:])
        return list(x_kfold_retVal, y_kfold_retVal)
