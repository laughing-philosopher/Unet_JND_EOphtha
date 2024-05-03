import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
# from patchify import patchify
from natsort import natsorted
# from patchify import unpatchify
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout, Lambda, Add, Multiply

from model import build_UNet
from functions import ce_loss, acc_met, sensitivity, specificity

# Patch dimension details
Img_Width = 48
Img_Height = 48
Img_Channels = 1

model = build_UNet((Img_Width, Img_Height, Img_Channels),(Img_Width, Img_Height, Img_Channels))
model.load_weights('UNet_JND_EOphtha.h5')

def processing(img):
    print(img.shape)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # The initial processing of the image
    img_g = img1[:,:,1]

    # Applying CLAHE as pre-processing step
    clahe = cv2.createCLAHE(clipLimit = 8, tileGridSize=(8,8))
    img_c = clahe.apply(img_g)

    [m,n] = img_c.shape

    gt = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gt = gt / 255.0
    gt = (gt > 0.5)

    tot = 0
    step = 8
    for i in range(0,m,step):
        for j in range(0,n,step):
            if((i+48)>(m-1) or (j+48)>(n-1)):
                pass
            else:
                tot = tot + 1

    threshold = 0.30 # increase this threshold if more false positives are coming
    cnt1 = 0
    cnt2 = 0
    iarr = []
    jarr = []
    patches_img = np.zeros((tot,48,48,1))
    patches_gt = np.zeros((tot,48,48,1))
    final_res = np.zeros((m,n))

    for i in range(0,m,step):
        for j in range(0,n,step):
            if((i+48)>(m-1) or (j+48)>(n-1)):
                pass
            else:
                # getting image patches
                patch_img = img_c[i:i+48,j:j+48]
                patch_img = np.expand_dims(patch_img, axis=-1)
                patch_img = np.expand_dims(patch_img, axis=0)
                patches_img[cnt1] = patch_img
                # getting gt patches
                patch_gt = gt[i:i+48,j:j+48]
                patch_gt = np.expand_dims(patch_gt, axis=-1)
                patch_gt = np.expand_dims(patch_gt, axis=0)
                patches_gt[cnt1] = patch_gt
                cnt1 = cnt1 + 1
                iarr.append(i)
                jarr.append(j)

    inter_res,_ = model.predict([patches_img,patches_gt],verbose=False)
    inter_res = (inter_res > threshold)
    for k in range(cnt1):
        final_res[iarr[k]:(iarr[k]+48),jarr[k]:(jarr[k]+48)] = final_res[iarr[k]:iarr[k]+48,jarr[k]:jarr[k]+48] + np.squeeze(inter_res[cnt2])
        cnt2 = cnt2 + 1

    psm_th2 = final_res / np.max(final_res)
    psm_th2 = (psm_th2 > 0.2).astype(float)

    return psm_th2