#!/usr/bin/env python

"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Pavan Mantripragada (mppavan@umd.edu) 
Masters in Robotics,
University of Maryland, College Park

Vishaal Kanna (vishaal@umd.edu) 
Masters in Robotics,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)
import keras.optimizers
import tensorflow as tf
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import Supervised_HomographyModel
from Network.Network import Unsupervised_HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
# from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
#from tqdm import tqdm
from Misc.TFSpatialTransformer import *
from Additional_functions.patch_gen import patch_generation

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain = ReadDirNames('./TxtFiles/DirNamesTrain.txt')
    DirNamesVal = ReadDirNames('./TxtFiles/DirNamesVal.txt')
    return DirNamesTrain, DirNamesVal

def SetupAll(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    DirNamesTrain, DirNamesVal = SetupDirNames(BasePath)

    ImageSize = [128, 128, 1]
    NumTrainSamples = len(DirNamesTrain)
    NumValSamples = len(DirNamesTrain)


    return DirNamesTrain, DirNamesVal, ImageSize, NumTrainSamples, NumValSamples


def GenerateBatch(BasePath, DirNamesTrain, ImageSize = 0, MiniBatchSize = 16, ModelType='Supervised'):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels
    """
    I1Batch = []
    LabelBatch = []
    Corners=[]
    Img_As = []
    Img_Bs = []
    imgs = []

    ImageNum = 0
    if ModelType == 'Supervised':
        # Generate random image
        while ImageNum < MiniBatchSize:
            RandIdx = random.randint(0, len(DirNamesTrain) - 1)

            RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'
            ImageNum += 1
            patch_size = 128

            img_orig = plt.imread(RandImageName)

            img_orig = np.float32(img_orig)

            if (len(img_orig.shape) == 3):
                img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
            else:
                img = img_orig

            img = (img - np.mean(img)) / 255
            img = cv2.resize(img, (320, 240))
            Img_data, Label, Img_A, Img_B, CA = patch_generation(img, patch_size)
            #maxv = Label.max()
            #minv = Label.min()
            #if abs(minv) > abs(maxv):
            #    maxv = abs(minv)
            #Label = Label / maxv

            I1Batch.append(Img_data)
            LabelBatch.append(Label)
        return I1Batch, LabelBatch

    else:
        # Generate random image
        while ImageNum < MiniBatchSize:
            RandIdx = random.randint(0, len(DirNamesTrain) - 1)

            RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'
            ImageNum += 1
            patch_size = 128

            img_orig = plt.imread(RandImageName)

            img_or = np.float32(img_orig)

            if (len(img_or.shape) == 3):
                img = cv2.cvtColor(img_or, cv2.COLOR_RGB2GRAY)
            else:
                img = img_orig

            # I1 = cv2.imread(RandImageName)
            img = (img - np.mean(img)) / 255
            img = cv2.resize(img, (320, 240))
            Img_data, Label, Img_A, Img_B, CA = patch_generation(img, patch_size)

            # Append All Images and Mask
            I1Batch.append(Img_data)
            LabelBatch.append(Label)
            Corners.append(CA)
            Img_As.append(Img_A)
            Img_Bs.append(Img_B)
            imgs.append(cv2.resize(img_orig, (320, 240)))
        return I1Batch, LabelBatch, Img_As, Img_Bs, Corners, imgs


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, ModelPath):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size: ' + str(MiniBatchSize))
    print('Number of Training Images: ' + str(NumTrainSamples))
    if ModelPath is not None:
        print('Loading latest checkpoint with the name ' + ModelPath)


def TrainOperation(DirNamesTrain, DirNamesVal, NumTrainSamples, NumValSamples, ImageSize,
                   NumEpochs, MiniBatchSize, DivTrain, BasePath, ModelType, ModelPath):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of data or for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    if ModelType == 'Supervised':
        # Predict output with forward pass
        model = Supervised_HomographyModel()
        if ModelPath is not None:
            model.load_weights(ModelPath)
            filepath = ModelPath
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
        else:
            filepath = "../Checkpoints/weights.best.hdf5"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]

        def euclidean_distance_loss(y_true, y_pred):
            return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))

        model.compile(loss=euclidean_distance_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
        training_loss = []
        validation_loss = []
        training_loss_epoch = []
        validation_loss_epoch = []
        for epoch in range(NumEpochs):
            print("\nStart of epoch %d" % (epoch+1,))
            tl, vl = 0, 0
            NumIterationsPerEpoch = 300 #int(NumTrainSamples / MiniBatchSize/ DivTrain)
            for PerEpochCounter in range(NumIterationsPerEpoch):
                I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, ImageSize, MiniBatchSize)
                history = model.fit(np.array(I1Batch), np.array(LabelBatch), batch_size=MiniBatchSize, callbacks=callbacks_list, verbose=0)
                training_loss.append(history.history['loss'])
                tl = np.add(tl, np.array(history.history['loss']))
                I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesVal, ImageSize,MiniBatchSize)
                results = model.evaluate(np.array(I1Batch), np.array(LabelBatch), verbose=0)
                validation_loss.append(results[0])
                vl = np.add(vl, np.array(results[0]))
            tl_epoch = tl/NumIterationsPerEpoch
            vl_epoch = vl / NumIterationsPerEpoch
            print('Training Loss: ', tl_epoch)
            print('Validation Loss: ', vl_epoch)
            training_loss_epoch.append(tl_epoch)
            validation_loss_epoch.append(vl_epoch)
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
        plt.show()
        plt.plot(training_loss_epoch)
        plt.plot(validation_loss_epoch)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
        plt.show()

    else:
        model = Unsupervised_HomographyModel(ImageSize = [128,128],MiniBatchSize=16)
        if ModelPath is not None:
            model.load_weights(ModelPath)
            filepath = ModelPath
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
        else:
            filepath = "../Checkpoints/weights.best.hdf5"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]

        model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

        training_loss = []
        validation_loss = []
        training_loss_epoch = []
        validation_loss_epoch = []
        for epoch in range(NumEpochs):
            print("\nStart of epoch %d" % (epoch+1,))
            tl, vl = 0, 0
            NumIterationsPerEpoch = 50
            for PerEpochCounter in range(NumIterationsPerEpoch):
                I1Batch, LabelBatch, Img_As, Img_Bs, Corners,_ = GenerateBatch(BasePath, DirNamesTrain, ImageSize, MiniBatchSize, ModelType)
                history = model.fit([np.array(I1Batch), np.array(Corners)], np.array(Img_Bs), batch_size=MiniBatchSize, callbacks=callbacks_list, verbose=0)
                training_loss.append(history.history['loss'])
                tl = np.add(tl, np.array(history.history['loss']))
                I1Batch, LabelBatch, Img_As, Img_Bs, Corners, _ = GenerateBatch(BasePath, DirNamesVal, ImageSize, MiniBatchSize, ModelType)
                results = model.evaluate([np.array(I1Batch), np.array(Corners)], np.array(Img_Bs), verbose=0)
                validation_loss.append(results[0])
                vl = np.add(vl, np.array(results[0]))
            tl_epoch = tl / NumIterationsPerEpoch
            vl_epoch = vl / NumIterationsPerEpoch
            print('Training Loss: ', tl_epoch)
            print('Validation Loss: ', vl_epoch)
            training_loss_epoch.append(tl_epoch)
            validation_loss_epoch.append(vl_epoch)
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
        plt.show()
        plt.plot(training_loss_epoch)
        plt.plot(validation_loss_epoch)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training Loss', 'Validation Loss'], loc='lower right')
        plt.show()


def main():
    """
    Inputs:
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../Data', help='Base path of images, Default:../Data')
    Parser.add_argument('--CheckPointPath', default= None, help='Path to save Weights, Default: None')
    Parser.add_argument('--ModelType', default='Supervised', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Supervised')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=16, help='Size of the MiniBatch to use, Default:16')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    ModelType = Args.ModelType
    ModelPath = Args.CheckPointPath

    # Setup all needed parameters including file reading
    DirNamesTrain, DirNamesVal, ImageSize, NumTrainSamples, NumValSamples = SetupAll(BasePath)

    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, ModelPath)

    TrainOperation(DirNamesTrain, DirNamesVal, NumTrainSamples, NumValSamples, ImageSize,
                   NumEpochs, MiniBatchSize, DivTrain, BasePath, ModelType, ModelPath)

if __name__ == '__main__':
    main()

