import sys

from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler

sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'pre')

import os
import cv2
import numpy as np
from enum import Enum
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from utilities.features import *

label_encoder = LabelEncoder()
scaler = StandardScaler()

"""
INITIAL PAThS
"""
MODEL_PATH = r'dataset\model\models'
SCALER_PATH = r'dataset\model\scalers'
FEATURE_PATH = r'dataset\model\features'
UNSEEN_PATH = r'dataset\images'
DATA_PATH = r'dataset\model'
DATASET_PATH = r'dataset\training'
AUG_PATH = r'dataset\augmented'
SEG_PATH = r'dataset\model'

"""
GENERAL CONSTANTS
"""
class Model(Enum):
    BaseModel       = 0
    ParticleSwarm   = 1
    AntColony       = 2
    ArtificialBee   = 3

class Disease(Enum):
    blb     =   0
    hlt     =   1
    rbl     =   2
    sbt     =   3

DISEASES = ['blb', 'hlt', 'rbl', 'sbt']

CLASSIFIER = SVC(C=10, kernel='rbf')
CORES = os.cpu_count() // 2
CORES = CORES if CORES // 2 >= os.cpu_count() else CORES + 1
PARAM_GRID = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto'] + [0.1, 1],
    # 'coef0': [0.0, 2.0],
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovr', 'ovo'],
    # 'shrinking': [True, False],
    'probability': [True, False],  # Add this line to control the randomness of the underlying implementation
    # 'random_state': [None, 0, 42]  # Add this line to control the seed of the random number generator
}
FOLDS = 1 # Amount of folds, KFOLD is automatically applied if FOLDS > 1
SHUFFLE = False # False to ensure replicability over all models

R_STATE = 42 # Select Random State to ensure replicablity
TEST_SIZE = 0.2 #  Percentage of test size
"""
OBJECTIVE FUNCTION
"""

def fitness(features, labels, subset):
    selected_features = features[:, subset]

    if FOLDS > 1:
        kfold = KFold(n_splits=FOLDS, shuffle=SHUFFLE, random_state=R_STATE)
        scores = cross_val_score(CLASSIFIER, selected_features, labels, cv=kfold)
        accuracy = scores.mean()
    else:
        X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=TEST_SIZE, random_state=R_STATE)
        CLASSIFIER.fit(X_train, y_train)
        y_pred = CLASSIFIER.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy
"""
FOR PRE-PROCESSING
"""
WIDTH, HEIGHT = 500, 500
FEAT_W = FEAT_H = 50
LTHRESHOLD = 128
DENOISE_KERNEL = (3, 3)
DENOISE_SIGMA = 0

SHARPEN_KERNEL = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=np.float32)

AUGMENTATIONS = [
    ('H_FLIP', lambda img: cv2.flip(img, 1)),
    ('V_FLIP', lambda img: cv2.flip(img, 0)),
    ('ROT90C', lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
    ('ROT180', lambda img: cv2.rotate(img, cv2.ROTATE_180)),
    ('ROT90O', lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ('CONINC', lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)),
    ('CONDEC', lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=0)),
    ('BR_INC', lambda img: cv2.convertScaleAbs(img, alpha=1, beta=50)),
    ('BR_DEC', lambda img: cv2.convertScaleAbs(img, alpha=1, beta=-50)),
    ('IMBLUR', lambda img: cv2.GaussianBlur(img, (5, 5), 0)),  # Apply Gaussian blur
    ('SHARPN', lambda img: cv2.filter2D(img, -1, SHARPEN_KERNEL)),  # Apply sharpening filter
]
"""
FOR FEATURE EXTRACTION
"""
# HOG (SHAPE) FEATURES
PPC = (8, 8)
CPB = (2, 2)
ORIENT = 9

# LBP (TEXTURE) FEATURES
RADIUS = 1
POINTS = 8 * RADIUS

# GLCM (TEXTURE) FEATURES
DISTANCE = 1
ANGLES = 0
LEVELS = 256

# COLOR HISTOGRAM FEATURES
BINS = 256
CHANNELS = (0, 1, 2)
RANGES = (0, 256)

# COLOR COHERANCE FEATURES
N_BINS = 8

FEATURES = {
    'SHP-HOG' : lambda image: getHOGFeatures(image, ORIENT, PPC, CPB), 
    'SHP-HOGIMG' : lambda image: getHOGImageFeatures(image, ORIENT, PPC, CPB), 
    'TXR-GLCM': lambda image: getGLCMFeatures(image, DISTANCE, ANGLES, LEVELS),
    'TXR-LBP' : lambda image: getLBPFeatures(image),
    'COL-HSV' : lambda image: getHSVFeatures(image),
    'COL-LAB' : lambda image: getLABFeatures(image),
    'COL-RGB' : lambda image: getRGBFeatures(image),
    'COL-STAT' : lambda image: getStatFeatures(image),
    'COL-COLHIST': lambda image: getColHistFeatures(image, BINS, CHANNELS, RANGES),
    'COL-HIST': lambda image: getHistFeatures(image, BINS),
    'COL-CCV': lambda image: getCCVFeatures(image, N_BINS)
}

GROUPED_FEATURES = {}
for feature, _ in FEATURES.items():
    prefix = feature.split('-')[0]  # Extract the prefix
    if prefix not in GROUPED_FEATURES:
        GROUPED_FEATURES[prefix] = []
    GROUPED_FEATURES[prefix].append(feature)