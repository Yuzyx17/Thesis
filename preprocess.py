import sys
sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')

import cv2
from pre.norm import *
from pre.segment import *
from utilities.util import displayImages, stopWait

features= []
labels= []

test = 0
disease = Disease.blb
index = 30

if test:
    displayImages(
        blb=segment_leaf(cv2.imread(f'dataset/captured/blb/{index}.jpg')),
        h=segment_leaf(cv2.imread(f'dataset/captured/hlt/{index}.jpg')),
        rb=segment_leaf(cv2.imread(f'dataset/captured/rbl/{index}.jpg')),
        sb=segment_leaf(cv2.imread(f'dataset/captured/sbt/{index}.jpg')),
    )
else:
    path = f'dataset/captured/{disease.value}/{index}.jpg'
    path = r'dataset\messenger\rbl\400714226_290472176661866_605769235617517701_n.jpg'
    img = cv2.imread(path)
    
    # displayChannels(img, alpha=1.25, lower=200, mask=True)
    displayImages(
        main=segment_leaf(img)
    )

stopWait()


