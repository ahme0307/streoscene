
from __future__ import print_function
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#import matplotlib.pyplot as plt
from data import load_train_data, load_test_data,plot_imagesT
import pdb
from skimage.io import imsave, imread
import cv2
import pickle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import data
#import pylab
import imageio
#import matplotlib.pyplot as plt
from  gen_data import load_image,random_batch,test_batch,load_images
from  get_resUnet import *
import params 
from os.path import splitext
from keras.utils import plot_model
from pathlib import Path
import dask.array as da
from dask.delayed import delayed
from dataloaders import *
 


def train_generator(imgs_train, imgs_mask_train, netpram):
    trainflow=Traindatagen.flow(imgs_train, imgs_mask_train, batch_size=netpram.batch_size)
    while True:
        x_batch,x_batch_right, y_batch,_ = trainflow.next()
        yield [x_batch,x_batch_right], y_batch



def valid_generator(imgs_test, imgs_mask_test, netprameval):
    validflow=Validdatagen.flow(imgs_test, imgs_mask_test, batch_size=netpram.batch_size)
    while True:
        x_batch,x_batch_right, y_batch,_ = validflow.next()
        yield [x_batch, x_batch_right],y_batch

def train(netpram,netprameval,d1,d2,imgs_train,imgs_test):
    model =YnetResNet(netpram)
    filename='YnetResNet2017_'+netpram.task+'_v'+d1+d2+'.hdf5'
    logdirs='%s%s'%(splitext(filename)[0],'logs')
    tensorboard = TensorBoard(log_dir=logdirs)
    callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=7,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath=filename,
                             save_best_only=True,
                             save_weights_only=True),
             tensorboard]





    history =model.fit_generator(generator=train_generator(imgs_train, imgs_mask_train, netpram),
                    steps_per_epoch=np.ceil(float(len(imgs_train)) / float(netpram.batch_size)),
                    epochs=netpram.nb_epoch,
                    max_queue_size=50, 
                    workers=4,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(imgs_test, imgs_mask_test, netprameval),
                    validation_steps=np.ceil(float(len(imgs_test)) / float(netpram.batch_size)))




if __name__ == "__main__":
    netparam=params.init() 
    netparameval=params.init(train=0) 
    x=[[1,3],[2,5],[4,8],[6,7]]
    data.create_train_data(netparam)
    data.create_test_data(netparam)
    for indx in range(0,4):
        imgs_train, imgs_mask_train=data.load_train_data()
        imgs_test,  imgs_mask_test =data.load_test_data()
        np.random.seed(1234)
        Traindatagen =CustomImageDataGenerator(netparam,training=1)
        Validdatagen= CustomImageDataGenerator(netparam,training=1)
        d1=str(x[indx][0])
        d2=str(x[indx][1])
        ids_train=[i for i, s in enumerate(imgs_mask_train) if 'instrument_dataset_'+d1  not in s and 'instrument_dataset_'+d2  not in s]
        ids_val = [i for i, s in enumerate(imgs_mask_test) if 'instrument_dataset_'+d1   in s or 'instrument_dataset_'+d2   in s]
        print("Data Information of Experiment: ", indx)
        print("------")
        print("  - No of Frames in Training set: %d" % len(ids_train))
        print("  - No of Frames in Test set %d" % len(ids_val))
        imgs_test=[imgs_test[img] for img in ids_val]
        imgs_mask_test=[imgs_mask_test[img] for img in ids_val]
        imgs_test=np.array(imgs_test)
        imgs_mask_test=np.array(imgs_mask_test)
        #imgs_mask_test =imgs_mask_test[ids_val_batch]
        imgs_train=[imgs_train[img] for img in ids_train]
        imgs_mask_train=[imgs_mask_train[img] for img in ids_train]
        imgs_train=np.array(imgs_train)
        imgs_mask_train=np.array(imgs_mask_train)
        train(netparam,netparameval,d1,d2,imgs_train,imgs_test)

    


