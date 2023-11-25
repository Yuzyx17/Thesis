import sys
sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')

import cv2
from pre.norm import *
from pre.segment import *
from utilities.util import displayChannels, displayImages, stopWait

features= []
labels= []

test = 1
disease = Disease.sb
index = 1
dataset = CAPTURED_PATH
if test:
    displayImages(
        blb=segment_leaf(cv2.imread(f'{dataset}/blb/{index}.jpg')),
        h=segment_leaf(cv2.imread(f'{dataset}/hlt/{index}.jpg')),
        rb=segment_leaf(cv2.imread(f'{dataset}/rb/{index}.jpg')),
        sb=segment_leaf(cv2.imread(f'{dataset}/sb/{index}.jpg')),
    )
else:
    path = f'{dataset}/{disease.name}/{index}.jpg'
    # path = r'dataset\google\blb1.jpg'
    img = cv2.imread(path)
    
    displayChannels(img, alpha=1.25, lower=200, mask=True)
    # displayImages(
    #     size=256,
    #     main=segment_leaf(img)
    # )

stopWait()


