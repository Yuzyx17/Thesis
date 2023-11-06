import cv2
import numpy as np
import joblib

from const import *

def rgbAsGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rgbAsLab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def labAsRgb(img):
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

def rgbAsHsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def hsvAsRgb(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def useMask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def useShape(img, x=-1, y=3):
    return img.reshape((x, y))

def loadDataSet(params=None):
    dataSet = ...
    if params is not None:
        dataSet = params

def displayImages(size=200, **imgs):
    for name, img in imgs.items():
        img = cv2.resize(img, (size, size))
        cv2.imshow(name, img)

def stopWait():
    cv2.waitKey()
    cv2.destroyAllWindows()

def saveSVC(model):
    # Define the file path where you want to save the model
    model_file_path = MODEL_PATH

    # Save the trained MSVM model to the specified file
    joblib.dump(model, model_file_path)