import sys
sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')
sys.path.append(r'utilities/features.py')

import cv2, os, time

from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from tqdm import tqdm

features= []
labels= []

# Pre process, Segment and Feature extract
for class_folder in os.listdir(DATASET_PATH):
    class_label = class_folder
    class_path = os.path.join(DATASET_PATH, class_folder)
    print(f"Class Label: {class_label}")
    if not os.path.exists(class_path):
        continue 

    # Loop through the images in the class folder
    for image_file in tqdm(os.listdir(class_path)):
        image_path = os.path.join(class_path, image_file)
        image = cv2.imread(image_path) 
        border_size = 10  # Adjust the border size as needed
        image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)

        # Pre processing
        test = useWWhiteBalance(image)
        test = cv2.medianBlur(test, ksize=3)

        lab = cv2.cvtColor(test, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(test, cv2.COLOR_RGB2HSV)
        l, a, B = cv2.split(lab)
        h, s, v = cv2.split(hsv)
        r, g, b = cv2.split(test)

        v = cv2.equalizeHist(v)
        g = cv2.equalizeHist(g)

        v = cv2.convertScaleAbs(v, alpha=1.25)
        g = cv2.convertScaleAbs(g, alpha=1.25)

        _, v = cv2.threshold(v, 195, 255, cv2.THRESH_BINARY)
        _, g = cv2.threshold(g, 195, 255, cv2.THRESH_BINARY)

        kernel_shape = cv2.MORPH_RECT  # You can use cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, or cv2.MORPH_CROSS
        kernel_size = (5, 5)  # Adjust the size as needed
        kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
        
        # Segmentation
        mask = cv2.bitwise_xor(v, s)
        mask = cv2.bitwise_xor(mask, g)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Thresholding
        _, mask = cv2.threshold(mask, 196, 255, cv2.THRESH_BINARY)
        image = cv2.bitwise_and(image, image, mask=mask)
        mask = cv2.resize(mask, (FEAT_W, FEAT_H))

        features.append(getFeatures(image))
        labels.append(class_label)

        # Apply augmentations
        # print(f'{class_label}:{image_file}')
        for augmentation_name, augmentation_fn in AUGMENTATIONS:
            # print(f'{class_label}:{image_file}-{augmentation_name}')

            aug_image = augmentation_fn(image)

            features.append(getFeatures(aug_image))
            labels.append(class_label)

features = np.array(features)
labels = np.array(labels)

unique_labels = np.unique(labels)
label_to_id = {label: i for i, label in enumerate(unique_labels)}
numerical_labels = np.array([label_to_id[label] for label in labels])

# Create a SimpleImputer to handle missing values (replace 'mean' with your preferred strategy)
imputer = SimpleImputer(strategy='mean')

# Apply imputation to your feature data
X = imputer.fit_transform(features)
# Initialize the scaler
scaler = StandardScaler()

# Fit on the imputed data
scaler.fit(X)
joblib.dump(scaler, f"{SCALER_PATH}/BaseModel.pkl")
# Transform the imputed data
X = scaler.transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, numerical_labels, test_size=0.2, random_state=42)

# Create an MSVM model with an RBF kernel
svm = CLASSIFIER

# Train the model on the training data
svm.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = svm.predict(X_test)

# Convert numerical labels back to original class labels
predicted_class_labels = [unique_labels[label] for label in Y_pred]

# Generate a classification report
report = classification_report(Y_test, Y_pred, target_names=unique_labels, zero_division='warn')

# Calculate the overall accuracy
overall_accuracy = accuracy_score(Y_test, Y_pred)
