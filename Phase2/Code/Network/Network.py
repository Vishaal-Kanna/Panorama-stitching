"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
from Misc.TFSpatialTransformer import transformer

# Don't generate pyc codes
sys.dont_write_bytecode = True

def Supervised_HomographyModel():
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    input_layer = tf.keras.Input(shape = (128, 128, 2))

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3 , activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    H4Pt = tf.keras.layers.Dense(8)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=H4Pt, name="TF_Functional_API")

    return model

def TensorDLT(H4pt, C4A, MiniBatchSize=16):
    Aux_M1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

    Aux_M2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)

    Aux_M3 = np.array([
        [0],
        [1],
        [0],
        [1],
        [0],
        [1],
        [0],
        [1]], dtype=np.float64)

    Aux_M4 = np.array([
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)

    Aux_M5 = np.array([
        [0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)

    Aux_M6 = np.array([
        [-1],
        [0],
        [-1],
        [0],
        [-1],
        [0],
        [-1],
        [0]], dtype=np.float64)

    Aux_M71 = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

    Aux_M72 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, -1, 0]], dtype=np.float64)

    Aux_M8 = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, -1]], dtype=np.float64)
    Aux_Mb = np.array([
        [0, -1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

    pts_tile_1 = tf.expand_dims(C4A, [2])

    pred_tile_H4 = tf.expand_dims(H4pt, [2])
    pred_pts_tile_2 = tf.add(pred_tile_H4, pts_tile_1)

    M1 = tf.tile(tf.expand_dims(tf.constant(Aux_M1, tf.float32), [0]), [MiniBatchSize, 1, 1])
    M2 = tf.tile(tf.expand_dims(tf.constant(Aux_M2, tf.float32), [0]), [MiniBatchSize, 1, 1])
    M3 = tf.tile(tf.expand_dims(tf.constant(Aux_M3, tf.float32), [0]), [MiniBatchSize, 1, 1])
    M4 = tf.tile(tf.expand_dims(tf.constant(Aux_M4, tf.float32), [0]), [MiniBatchSize, 1, 1])
    M5 = tf.tile(tf.expand_dims(tf.constant(Aux_M5, tf.float32), [0]), [MiniBatchSize, 1, 1])
    M6 = tf.tile(tf.expand_dims(tf.constant(Aux_M6, tf.float32), [0]), [MiniBatchSize, 1, 1])
    M71 = tf.tile(tf.expand_dims(tf.constant(Aux_M71, tf.float32), [0]), [MiniBatchSize, 1, 1])
    M72 = tf.tile(tf.expand_dims(tf.constant(Aux_M72, tf.float32), [0]), [MiniBatchSize, 1, 1])
    M8 = tf.tile(tf.expand_dims(tf.constant(Aux_M8, tf.float32), [0]), [MiniBatchSize, 1, 1])
    Mb = tf.tile(tf.expand_dims(tf.constant(Aux_Mb, tf.float32), [0]), [MiniBatchSize, 1, 1])

    A1 = tf.matmul(M1, pts_tile_1)
    A2 = tf.matmul(M2, pts_tile_1)
    A3 = M3
    A4 = tf.matmul(M4, pts_tile_1)
    A5 = tf.matmul(M5, pts_tile_1)
    A6 = M6
    A7 = tf.matmul(M71, pred_pts_tile_2) * tf.matmul(M72, pts_tile_1)
    A8 = tf.matmul(M71, pred_pts_tile_2) * tf.matmul(M8, pts_tile_1)

    A_mat = tf.transpose(tf.stack([tf.reshape(A1, [-1, 8]), tf.reshape(A2, [-1, 8]), tf.reshape(A3, [-1, 8]), tf.reshape(A4, [-1, 8]), tf.reshape(A5, [-1, 8]), tf.reshape(A6, [-1, 8]), \
                                   tf.reshape(A7, [-1, 8]), tf.reshape(A8, [-1, 8])], axis=1), perm=[0, 2, 1])

    B_mat = tf.matmul(Mb, pred_pts_tile_2)

    H_8el = tf.linalg.solve(A_mat, B_mat)

    h_ones = tf.ones([MiniBatchSize, 1, 1])
    H_9el = tf.concat([H_8el, h_ones], 1)
    H_flat = tf.reshape(H_9el, [-1, 9])
    H_mat = tf.reshape(H_flat, [-1, 3, 3])

    return H_mat


def Unsupervised_HomographyModel(ImageSize = [128,128],MiniBatchSize=16):
    """
    Inputs:
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    input_layer = tf.keras.Input(shape = (128, 128, 2))
    C4A = tf.keras.Input(shape = (8))

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3 , activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3 , activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    H4Pt = tf.keras.layers.Dense(8)(x)

    H_mat = TensorDLT(H4Pt, C4A, MiniBatchSize)

    h = ImageSize[1]
    w = ImageSize[0]

    M = np.array([[w/2.0, 0., w/2.0], [0., h/2.0, h/2.0], [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [MiniBatchSize, 1, 1])

    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [MiniBatchSize, 1, 1])

    H_mat = tf.linalg.matmul(tf.linalg.matmul(M_tile_inv, H_mat), M_tile)
    size_out = (h, w)

    ImgA = tf.slice(input_layer, [0, 0, 0, 0], [MiniBatchSize, 128, 128, 1])

    warped_images, _ = transformer(ImgA, H_mat, size_out)
    warped_gray_images = tf.reduce_mean(warped_images, 3)
    pred_I2_flat = warped_gray_images
    Img_B_predicted = tf.reshape(pred_I2_flat, [MiniBatchSize, 128, 128, 1])


    model = tf.keras.Model(inputs=[input_layer, C4A], outputs=Img_B_predicted, name="TF_Functional_API_1")

    return model