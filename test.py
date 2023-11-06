import sys
sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')

import cv2
from pre.norm import *
from pre.segment import *
from utilities.util import displayImages, stopWait
from utilities.features import getHOGFeatures, getLBPFeatures
from lib.classifier import *

image_path = r'dataset/images/blast/2.jpg'
image = cv2.imread(image_path)
image = useResize(image)

test= useWWhiteBalance(image)
test = useCLAHE(test)

test = useDenoise(test)
test = useCustomThreshold(test)

features = getHOGFeatures(image) + getLBPFeatures(image)

print(fitness_function(features))

displayImages(
    Main=image,
    W=test,
)

stopWait()