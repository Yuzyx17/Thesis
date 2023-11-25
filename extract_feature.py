import cv2, os

from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from tqdm import tqdm

from utilities.util import displayImages, getFeatures

features = []
labels = []
path = TRAINING_PATH
output_path = AUGMENTED_PATH if path is PHILRICE_PATH else SEGMENTED_PATH
output_path = r'dataset\finalized\segmented'
save = False
augment = True if path is CAPTURED_PATH or path is PHILRICE_PATH else False
augment = True
# Loop through the class folders
for class_folder in os.listdir(path):
    class_label = class_folder
    class_path = os.path.join(path, class_folder)
    print(f"Class Label: {class_label}")
    if not os.path.exists(class_path):
        continue 

    # Loop through the images in the class folder
    for image_file in tqdm(os.listdir(class_path)):
        image_path = os.path.join(class_path, image_file)
        image = cv2.imread(image_path) 
        # if path is not output_path:
        image = segment_leaf(image)
        # seg_image = cv2.resize(seg_image, (FEAT_W, FEAT_H))
        if save:
            seg_image = image
            cv2.imwrite(os.path.join(output_path, class_folder, image_file), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # else:
        image = cv2.resize(image, (FEAT_W, FEAT_H))
        
        features.append(getFeatures(image))
        labels.append(class_label)

        if augment:
            for augmentation_name, augmentation_fn in AUGMENTATIONS:
                aug_image = augmentation_fn(image)
                # if path is not output_path:
                # seg_aug_image = segment_leaf(aug_image)

                # seg_aug_image = cv2.resize(seg_aug_image, (FEAT_W, FEAT_H))
                if save:
                    seg_aug_image = augmentation_fn(seg_image)
                    cv2.imwrite(os.path.join(output_path, class_folder, f"{augmentation_name}-{image_file}"), seg_aug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # else:
                # seg_aug_image = cv2.resize(image, (FEAT_W, FEAT_H))
                features.append(getFeatures(aug_image))
                labels.append(class_label)

features = np.array(features)
labels = np.array(labels)

print(f"Saving Features ({features.shape})")
np.save(f"{DATA_PATH}/features.npy", features)
np.save(f"{DATA_PATH}/labels.npy", labels)
print("Features Saved")