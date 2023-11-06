import sys
sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')

import cv2
import os
from pre.norm import *
from pre.segment import *
from utilities.util import displayImages, stopWait
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern

import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

image_path = r'dataset\captured\rice-tungro\2.jpg'
image = cv2.imread(image_path)

dataset_dir = r'dataset\captured'
augmented_dir = r'dataset\augmented\captured'
model_file_path = r'dataset\model'
features= []
labels= []

displayImages(
    Main=image,
    blb=segment_leaf(cv2.imread(r'dataset\captured\bacterial-leaf-blight\1.jpg')),
    bs=segment_leaf(cv2.imread(r'dataset\captured\brown-spot\1.jpg')),
    h=segment_leaf(cv2.imread(r'dataset\captured\healthy\1.jpg')),
    rb=segment_leaf(cv2.imread(r'dataset\captured\rice-blast\1.jpg')),
    rt=segment_leaf(cv2.imread(r'dataset\captured\rice-tungro\1.jpg')),
    sb=segment_leaf(cv2.imread(r'dataset\captured\sheath-blight\1.jpg')),
)

stopWait()


