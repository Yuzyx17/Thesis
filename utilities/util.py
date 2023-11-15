import sys
import cv2
import numpy as np
import joblib

from const import *
from skimage.feature import graycomatrix

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

def displayImages(size=175, **imgs):
    for name, img in imgs.items():
        img = cv2.resize(img, (size, size))
        cv2.imshow(name, img)

def displayChannels(image, size=150, alpha=1, upper=255, lower=127, eq=True, mask=False):
    rgb = cv2.medianBlur(image, ksize=3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    l, a, B = cv2.split(lab)
    h, s, v = cv2.split(hsv)
    r, g, c = cv2.split(rgb)
    
    l = cv2.equalizeHist(l)
    a = cv2.equalizeHist(a)
    B = cv2.equalizeHist(B)
    h = cv2.equalizeHist(h)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    c = cv2.equalizeHist(c)
    
    # a = 255-a
    B = 255-B
    l = 255-l

    l = cv2.convertScaleAbs(l, alpha=alpha)
    a = cv2.convertScaleAbs(a, alpha=alpha)
    B = cv2.convertScaleAbs(B, alpha=alpha)
    h = cv2.convertScaleAbs(h, alpha=alpha)
    s = cv2.convertScaleAbs(s, alpha=alpha)
    v = cv2.convertScaleAbs(v, alpha=alpha)
    r = cv2.convertScaleAbs(r, alpha=alpha)
    g = cv2.convertScaleAbs(g, alpha=2.5)
    c = cv2.convertScaleAbs(c, alpha=alpha)

    _, l = cv2.threshold(l, lower, upper, cv2.THRESH_BINARY)
    _, a = cv2.threshold(a, lower, upper, cv2.THRESH_BINARY)
    _, B = cv2.threshold(B, lower, upper, cv2.THRESH_BINARY)
    _, h = cv2.threshold(h, lower, upper, cv2.THRESH_BINARY)
    _, s = cv2.threshold(s, lower, upper, cv2.THRESH_BINARY)
    _, v = cv2.threshold(v, lower, upper, cv2.THRESH_BINARY)
    _, r = cv2.threshold(r, lower, upper, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, lower, upper, cv2.THRESH_BINARY)
    _, c = cv2.threshold(c, lower, upper, cv2.THRESH_BINARY)

    # Define the shape of the kernel (e.g., a square)
    kernel_shape = cv2.MORPH_RECT  # You can use cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, or cv2.MORPH_CROSS

    # Define the size of the kernel (width and height)
    kernel_size = (5, 5)  # Adjust the size as needed

    # Create the kernel
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    
    leaf = cv2.bitwise_or(h, g)
    leaf = cv2.bitwise_and(leaf, s)
    leaf = cv2.dilate(leaf, kernel, iterations=2)

    dise = cv2.bitwise_and(a, h)
    dise = cv2.bitwise_or(dise, B)

    imask = cv2.bitwise_xor(dise, leaf)

    if mask:
        l = cv2.bitwise_and(image, image, mask=l)
        a = cv2.bitwise_and(image, image, mask=a)
        B = cv2.bitwise_and(image, image, mask=B)
        h = cv2.bitwise_and(image, image, mask=h)
        s = cv2.bitwise_and(image, image, mask=s)
        v = cv2.bitwise_and(image, image, mask=v)
        r = cv2.bitwise_and(image, image, mask=r)
        g = cv2.bitwise_and(image, image, mask=g)
        c = cv2.bitwise_and(image, image, mask=c)

    masked = cv2.bitwise_and(image, image, mask=imask)
    leaf = cv2.bitwise_and(image, image, mask=leaf)
    dise = cv2.bitwise_and(image, image, mask=dise)

    displayImages(
        Main=masked,
        l=l,
        a=a,
        B=B,
        y=h,
        s=s,
        v=v,
        r=r,
        g=g,
        c=c,
        p=dise,
        u=leaf,
    )


def progressBar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()
    
def stopWait():
    cv2.waitKey()
    cv2.destroyAllWindows()

def saveSVC(model, name="base"):
    print("Model Saving")
    # Define the file path where you want to save the model
    model_file_path = f"{MODEL_PATH}/{name}.joblib"

    # Save the trained MSVM model to the specified file
    joblib.dump(model, model_file_path)
    print("Model Saved")

