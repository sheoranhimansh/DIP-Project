import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import io
from PIL import Image


train_path = "imagedataset/"

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
def Hog_feature(image):
    hog = cv2.HOGDescriptor()
    h = hog.compute(image)
    return h.flatten()


global_features = []

for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    for file in os.listdir(dir):
        temp = file.split('.')
        input_file = dir +'/'+file
        image = cv2.imread(input_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (250, 250)) 
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        labels.append(current_label)
        global_features.append(global_feature)
