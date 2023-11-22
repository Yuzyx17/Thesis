import sys
import cv2
import numpy as np
import joblib, functools

from utilities.const import *

def displayImages(size=175, **imgs):
    for model, img in imgs.items():
        img = cv2.resize(img, (size, size))
        cv2.imshow(model, img)

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

def stopWait():
    cv2.waitKey()
    cv2.destroyAllWindows()

# def predictImage(image, model: Model):
    # classifier = joblib.load(f"{MODEL_PATH}/{model.name}.joblib")
    # scaler = joblib.load(f"{SCALER_PATH}/{model.name}.pkl")
    # encoder = joblib.load(r'dataset\model\encoder.joblib')

    # image = cv2.imread(image) 
    # image = segment_leaf(image)
    # X = [getFeatures(image)]
    # X = scaler.transform(X)

    # if model is not Model.BaseModel:
    #     selected_features = np.load(f"{FEATURE_PATH}/{model.name}.npy")
    #     X = X[:,selected_features]
    
    # prediction = classifier.predict_proba(X)
    # print(encoder.classes_)

    # return prediction

def saveModel(classifier, model, subset=None):
    print("Model Saving")
    joblib.dump(classifier, f"{MODEL_PATH}/{model.name}.joblib")
    joblib.dump(scaler, f"{SCALER_PATH}/{model.name}.pkl")
    if subset is not None:
        np.save(f"{FEATURE_PATH}/{model.name}.npy", subset)
    print("Model Saved")

def getFeatures(image):
    features = []
    for _, feature_func in FEATURES.items():
        features.extend(feature_func(image))
    return np.array(features)
