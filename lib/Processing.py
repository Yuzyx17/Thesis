import cv2
import joblib
from const import *

def segment(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    v = cv2.equalizeHist(v)
    v = cv2.convertScaleAbs(v, alpha=1.25)
    _, v = cv2.threshold(v, LB, UB, cv2.THRESH_BINARY)
    
    # Thresholding based segmentation
    mask = cv2.bitwise_or(s, v)
    _, mask = cv2.threshold(mask, LB, UB, cv2.THRESH_BINARY)
    image = cv2.bitwise_and(image, image, mask=mask)

    return image

def predictImage(image, model: ModelType):
    classifier = joblib.load(f"{MODEL_PATH}/{model.name}.joblib")
    scaler = joblib.load(f"{SCALER_PATH}/{model.name}.pkl")
    encoder = joblib.load(r'dataset\model\encoder.joblib')

    image = cv2.imread(image) 
    image = segment(image)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    X = [extractFeatures(image)]
    X = scaler.transform(X)

    if model is not ModelType.BaseModel:
        selected_features = np.load(f"{FEATURE_PATH}/{model.name}.npy")
        X = X[:,selected_features]
    
    prediction = classifier.predict_proba(X)
    print(encoder.classes_)

    return prediction

def loadImages(dataset_path):
    features = []
    labels = []
    # Loop through the class folders
    for class_folder in os.listdir(dataset_path):
        class_label = class_folder
        class_path = os.path.join(dataset_path, class_folder)

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path) 
            seg_image = segment(image)
            seg_image = cv2.resize(seg_image, (WIDTH, HEIGHT))
            
            feature = extractFeatures(seg_image)
            features.append(feature)
            labels.append(class_label)

            for _, augment in AUGMENTATIONS:
                aug_image = augment(image)
                aug_image = segment(aug_image)
                aug_image = cv2.resize(aug_image, (WIDTH, HEIGHT))

                feature = extractFeatures(aug_image)
                features.append(feature)
                labels.append(class_label)
    
    features = np.array(features)
    labels = np.array(labels)
    np.save(f"{DATA_PATH}/features.npy", features)
    np.save(f"{DATA_PATH}/labels.npy", labels)
    return features, labels

def preLoadImages():
    features = np.load(f"{DATA_PATH}/features.npy")
    labels = np.load(f"{DATA_PATH}/labels.npy")

    return features, labels

def extractFeatures(image):
    features = []
    for _, feature_func in FEATURES.items():
        features.extend(feature_func(image))
    return np.array(features)
