import sys
sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')

import cv2
from pre.norm import *
from pre.segment import *
from utilities.util import displayImages, stopWait

dataset_dir = r'dataset\captured'
augmented_dir = r'dataset\augmented\captured'
model_file_path = r'dataset\model'
features= []
labels= []

test = 0
disease = 2
index = 7
disease = 'blb' if disease==1 else ('hlt' if disease==2 else ('rbl' if disease==3 else 'sbt'))

if test:
    displayImages(
        blb=segment_leaf(cv2.imread(f'dataset/captured/blb/{index}.jpg')),
        h=segment_leaf(cv2.imread(f'dataset/captured/hlt/{index}.jpg')),
        rb=segment_leaf(cv2.imread(f'dataset/captured/rbl/{index}.jpg')),
        sb=segment_leaf(cv2.imread(f'dataset/captured/sbt/{index}.jpg')),
    )
else:
    img = cv2.imread(f'dataset/captured/{disease}/{index}.jpg')
    displayChannels(img, alpha=1.25, lower=200, mask=True)

stopWait()


