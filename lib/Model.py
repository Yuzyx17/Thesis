import joblib
from sklearn.metrics import classification_report
from lib.Processing import extractFeatures, segment
from lib.WrapperACO import WrapperACO
from lib.WrapperPSO import WrapperPSO
from const import *

class Model():
    def __init__(self, model: ModelType, features=None, labels=None):
        self.model = model
        self.classifier = None
        self.accuracy = 0.0
        self.report = None
        self.solution = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

        if features is not None and labels is not None:
            Y = self.encoder.fit_transform(labels)
            X_train, X_test, self.Y_train, self.Y_test = train_test_split(features, Y, test_size=TEST_SIZE, random_state=R_STATE)
            self.X_train = self.scaler.fit_transform(X_train)
            self.X_test = self.scaler.transform(X_test)
    
    def save(self):
        joblib.dump(self.classifier, f"{MODEL_PATH}/{self.model.name}.joblib")
        joblib.dump(self.scaler, f"{SCALER_PATH}/{self.model.name}.pkl")
        joblib.dump(self.encoder, f"{DATA_PATH}/encoder.pkl")
        if self.solution is not None:
            np.save(f"{FEATURE_PATH}/{self.model.name}.npy", self.solution)

    def load(self):
        self.classifier = joblib.load(f"{MODEL_PATH}/{self.model.name}.joblib")
        self.scaler = joblib.load(f"{SCALER_PATH}/{self.model.name}.pkl")
        self.encoder = joblib.load(f"{DATA_PATH}/encoder.pkl")
        if self.model is not ModelType.BaseModel:
            self.solution = np.load(f"{FEATURE_PATH}/{self.model.name}.npy")

    def create(self):
        match self.model:
            case ModelType.AntColony:
                fitness_function = lambda subset: fitness_cv(self.X_train, self.Y_train, subset)
                fit_accuracy = fitness_function(np.arange(0, self.X_train.shape[1]))
                aco = WrapperACO(fitness_function,
                                self.X_train.shape[1], 
                                ants=15, 
                                iterations=20, 
                                rho=0.1, 
                                Q=.75, 
                                debug=1, 
                                accuracy=fit_accuracy, 
                                parrallel=True, 
                                cores=6)                
                self.solution, _ = aco.optimize()

            case ModelType.ParticleSwarm:
                fitness_function = lambda subset: fitness_pso(self.X_train, self.Y_train, subset)
                pso = WrapperPSO(fitness_function, 
                                 self.X_train.shape[1], 
                                 particles=1, 
                                 iterations=1)
                self.solution = pso.optimize()
                
        X_train = self.X_train[:, self.solution] if self.solution is not None else self.X_train
        X_test = self.X_test[:, self.solution] if self.solution is not None else self.X_test

        svm = SVC(C=10, kernel='rbf', probability=True)
        svm.fit(X_train, self.Y_train)

        Y_pred = svm.predict(X_test)
        self.report = classification_report(self.Y_test, Y_pred, target_names=self.encoder.classes_, zero_division='warn', output_dict=True)
        self.accuracy = accuracy_score(self.Y_test, Y_pred)
        self.classifier = svm

    def predict(self, image):
        image = cv2.imread(image) 
        image = segment(image)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        X = [extractFeatures(image)]
        X = self.scaler.transform(X)

        if self.model is not ModelType.BaseModel:
            X = X[:, self.solution]
        
        prediction = self.classifier.predict_proba(X)
        predicted = self.classifier.predict(X)
        
        predictions = {label:probability for (label, probability) in zip(self.encoder.classes_, prediction[0])}
        predictions['predicted'] = self.encoder.classes_[predicted[0]] 
        return predictions