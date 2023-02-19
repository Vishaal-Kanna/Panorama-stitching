#!/usr/bin/env python

"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


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

import tensorflow as tf
import cv2
import os
import sys
import glob
#import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import Supervised_HomographyModel
from Network.Network import Unsupervised_HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Train import GenerateBatch
import numpy as np
import time
import argparse
import shutil
#from StringIO import StringIO
import string
import math as m
#from tqdm import tqdm
from Misc.TFSpatialTransformer import *


# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTest = ReadDirNames('./TxtFiles/DirNamesTest.txt')

    return DirNamesTest

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
    DirNamesTrain = SetupDirNames(BasePath)

    ImageSize = [128, 128, 1]
    NumTrainSamples = len(DirNamesTrain)


    return DirNamesTrain, ImageSize, NumTrainSamples

def TestOperation(BasePath, ModelPath, DirNamesTest, ModelType ='Supervised'):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    if ModelType == 'Supervised':
        model = Supervised_HomographyModel()

        model.load_weights(ModelPath)
        def euclidean_distance_loss(y_true, y_pred):
            return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))
        model.compile(loss=euclidean_distance_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

        I1Batch, LabelBatch, Img_As, Img_Bs, Corners,imgs = GenerateBatch(BasePath, DirNamesTest, MiniBatchSize=54, ModelType = 'Unsupervised')

        results = model.evaluate(np.array(I1Batch), np.array(LabelBatch), verbose=0)
        print("Error during Testing with Supervised Model:", results[0])

    else:
        model = Unsupervised_HomographyModel(MiniBatchSize = 16)

        model.load_weights(ModelPath)
        model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
        I1Batch, LabelBatch, Img_As, Img_Bs, Corners,imgs = GenerateBatch(BasePath, DirNamesTest, MiniBatchSize = 16, ModelType = ModelType)

        results = model.evaluate([np.array(I1Batch),np.array(Corners)], np.array(Img_Bs), verbose=0)
        print("Error during Testing with Unsupervised Model:", results[0])

def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/weights.best.hdf5', help='Path to load latest model from, Default:../Checkpoints/weights.best.hdf5')
    Parser.add_argument('--BasePath', dest='BasePath', default='../Data', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--ModelType', default='Supervised', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Supervised')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    DirNamesTest, ImageSize, NumTrainSamples = SetupAll(BasePath)

    TestOperation(BasePath, ModelPath, DirNamesTest, ModelType)

     
if __name__ == '__main__':
    main()
 
