import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from lib.Processing import extractFeatures, segment
from lib.WrapperABC import WrapperABC
from lib.WrapperACO import WrapperACO
from lib.WrapperPSO import WrapperPSO
from const import *

class Model():
    def __init__(self, model: ModelType, features=None, labels=None):
        self.model = model
        self.accuracy = 0.0
        self.classifier = None
        self.report = None
        self.solution = None
        self.confusion = None
        self.metrics = None
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
        print(f"Starting Training")
        match self.model:
            case ModelType.AntColony:
                fitness_function = lambda subset: fitness_cv(self.X_train, self.Y_train, subset) if FOLDS > 1 else fitness(self.X_train, self.Y_train, subset)
                fit_accuracy = fitness_function(np.arange(0, self.X_train.shape[1]))
                aco = WrapperACO(fitness_function,
                                self.X_train.shape[1], 
                                ants=20, 
                                iterations=50, 
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

            case ModelType.ArtificialBee:
                fitness_function = lambda subset: fitness_cv(self.X_train, self.Y_train, subset) if FOLDS > 1 else fitness(self.X_train, self.Y_train, subset)
                abc = WrapperABC(fitness_function, self.X_train.shape[1], bees=10, iterations=10)
                self.solution = abc.optimize()

        X_train = self.X_train[:, self.solution] if self.solution is not None else self.X_train
        X_test = self.X_test[:, self.solution] if self.solution is not None else self.X_test

        svm = SVC(C=10, kernel='rbf', probability=True)
        svm.fit(X_train, self.Y_train)

        Y_pred = svm.predict(X_test)
        self.report = classification_report(self.Y_test, Y_pred, target_names=self.encoder.classes_, zero_division='warn', output_dict=True)
        self.accuracy = accuracy_score(self.Y_test, Y_pred)
        self.classifier = svm

    def retestModel(self):
        X_test = self.X_test[:, self.solution] if self.solution is not None else self.X_test

        Y_pred = self.classifier.predict(X_test)
        report = classification_report(self.Y_test, Y_pred, target_names=self.encoder.classes_, zero_division='warn', output_dict=True)
        accuracy = accuracy_score(self.Y_test, Y_pred)

        print(f"{report} Accuracy: {accuracy}")

    def obtainMetrics(self, test=None):
        X_test, Y_test = test if test is not None else (self.X_test, self.Y_test)
        X_test = self.scaler.transform(X_test) if test is not None else Y_test
        X_test = X_test[:, self.solution] if self.solution is not None else X_test 
        Y_test = self.encoder.transform(Y_test) if test is not None else Y_test

        Y_pred = self.classifier.predict(X_test)
        report = classification_report(Y_test, Y_pred, target_names=self.encoder.classes_, zero_division='warn', output_dict=True)
        accuracy = accuracy_score(Y_test, Y_pred)
        confusion = confusion_matrix(Y_test, Y_pred, normalize='all')

        FP = confusion.sum(axis=0) - np.diag(confusion)  
        FN = confusion.sum(axis=1) - np.diag(confusion)
        TP = np.diag(confusion)
        TN = confusion.sum() - (FP + FN + TP)

        # Overall accuracy
        PRECISION = TP/(TP+FP)
        RECALL = TP/(TP+FN)
        F1 = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
        ACCURACY = (TP+TN)/(TP+FP+FN+TN)

        macro_precision = precision_score(Y_test, Y_pred, average='macro')
        macro_recall = recall_score(Y_test, Y_pred, average='macro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')

        micro_precision = precision_score(Y_test, Y_pred, average='micro')
        micro_recall = recall_score(Y_test, Y_pred, average='micro')
        micro_f1 = f1_score(Y_test, Y_pred, average='micro')

        weighted_precision = precision_score(Y_test, Y_pred, average='weighted')
        weighted_recall = recall_score(Y_test, Y_pred, average='weighted')
        weighted_f1 = f1_score(Y_test, Y_pred, average='weighted')

        stack = np.array((PRECISION, RECALL, F1, ACCURACY))
        labels = ['precision', 'recall', 'f1', 'accuracy']
        self.metrics = {
            'blb'   :   {v:c for c,v in zip(stack[:, 0], labels)},
            'hlt'   :   {v:c for c,v in zip(stack[:, 1], labels)},
            'rb'    :   {v:c for c,v in zip(stack[:, 2], labels)},
            'sb'    :   {v:c for c,v in zip(stack[:, 3], labels)},
            'macro' :   {
                'precision':    macro_precision,
                'recall':       macro_recall,
                'f1':           macro_f1,
            },
            'micro' :   {
                'precision':    micro_precision,
                'recall':       micro_recall,
                'f1':           micro_f1,
            },
            'weighted': {
                'precision':    weighted_precision,
                'recall':       weighted_recall,
                'f1':           weighted_f1,
            },
            'accuracy': accuracy,
            'report': report
        }

        return self.metrics
    
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
    
    def obtainConfusion(self, test=None):
        X_test, Y_test, N_test = test if test is not None else (self.X_test, self.Y_test)
        X_test = self.scaler.transform(X_test) if test is not None else Y_test
        X_test = X_test[:, self.solution] if self.solution is not None else X_test 
        Y_test = self.encoder.transform(Y_test) if test is not None else Y_test
        Y_pred = self.classifier.predict(X_test)
        
        print(f"Image\t\tPredicted\tActual")
        for i in range(len(Y_pred)):
            print(f"{N_test[i]}\t\t{self.encoder.classes_[Y_pred[i]]}\t\t{self.encoder.classes_[Y_test[i]]}")

        confusion = confusion_matrix(Y_test, Y_pred)
        norm_confusion = confusion_matrix(Y_test, Y_pred)
        FP = norm_confusion.sum(axis=0) - np.diag(norm_confusion)  
        FN = norm_confusion.sum(axis=1) - np.diag(norm_confusion)
        TP = np.diag(norm_confusion)
        TN = norm_confusion.sum() - (FP + FN + TP)

        # Overall accuracy
        PRECISION = TP/(TP+FP)
        RECALL = TP/(TP+FN)
        F1 = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
        ACCURACY = (TP+TN)/(TP+FP+FN+TN)

        data_frame = pd.DataFrame(confusion, index=['blb', 'hlt', 'sb', 'rb'], columns=['blb', 'hlt', 'sb', 'rb'])
        plt.figure(figsize=(5, 4))
        sb.heatmap(data_frame, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        confusion = confusion.reshape((4, 4))
        print(confusion)
        for i in range(len(confusion[0])):
            print(self.encoder.classes_[i])
            fn = 0
            fp = 0
            for j in range(len(confusion[i])):
                if i != j:
                    fn += confusion[i][j]
                    fp += confusion[j][i]
            tp = confusion[i][i]
            tn = confusion.sum() - (confusion[i][i] + fn + fp)

            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = 2 * ((precision*recall)/(precision+recall))
            accuracy = (tp+tn)/(tp+fp+fn+tn)
            print(f"TP {tp}\nTN {tn}\nFN {fn}\nFP {fp}")
            print(f"Precision {precision}\nRecall {recall}\nF1-Score {f1}\nAccuracy {accuracy}\n")

        