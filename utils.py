import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, precision_recall_curve, auc, accuracy_score, classification_report
from keras.applications.mobilenet_v3 import preprocess_input
from homemade_dataloader import HomemadeDataloader

class MatrixPlotter:
    def __init__(self):
        # Dictionary to store the metrics for the specified dataset
        self.metrics = {
            'sensitivity': 0,
            'specificity': 0,
            'mean_avg_precision': 0,
            'accuracy': 0,
            'balanced_accuracy': 0,
            'f1_score': 0
        }
        self.confusion_matrix = None  # Store confusion matrix for the dataset
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']  

    def calculate_metrics(self, model, data_generator):
        y_true, y_pred, y_pred_proba = [], [], []

        # Get predictions on the test set
        for batch_x, batch_y in data_generator:
            preds = model.predict(batch_x)
            y_pred.extend(np.argmax(preds, axis=1))
            y_pred_proba.extend(np.max(preds, axis=1))
            y_true.extend(np.argmax(batch_y, axis=1))

        print(y_pred_proba)
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrix = cm

        # Calculate confusion matrix elements for sensitivity and specificity
        tp = np.diag(cm)  # True positives per class
        fn = np.sum(cm, axis=1) - tp  # False negatives per class
        fp = np.sum(cm, axis=0) - tp  # False positives per class
        tn = cm.sum() - (fp + fn + tp)  # True negatives per class

        # Calculate metrics
        sensitivity = np.mean(tp / (tp + fn)) if np.any(tp + fn) else 0
        specificity = np.mean(tn / (tn + fp)) if np.any(tn + fp) else 0
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Mean Average Precision (mAP)
        precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba, pos_label=np.argmax(y_true))
        mean_avg_precision = auc(recalls, precisions)

        # Store metrics
        self.metrics['sensitivity'] = sensitivity
        self.metrics['specificity'] = specificity
        self.metrics['mean_avg_precision'] = mean_avg_precision
        self.metrics['accuracy'] = accuracy
        self.metrics['balanced_accuracy'] = balanced_acc
        self.metrics['f1_score'] = f1
        print(f"{classification_report(y_true, y_pred, target_names=self.class_names, digits=4)}")

    def print_metrics(self):
        # Print single-value metrics
        print("Test Set Metrics:")
        print(f"Sensitivity: {self.metrics['sensitivity']:.4f}")
        print(f"Specificity: {self.metrics['specificity']:.4f}")
        print(f"Mean Average Precision (mAP): {self.metrics['mean_avg_precision']:.4f}")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {self.metrics['balanced_accuracy']:.4f}")
        print(f"F1 Score: {self.metrics['f1_score']:.4f}")
        

    def plot_precision_recall_curve(self, data_generator):
        # Plot the Precision-Recall Curve
        y_true, y_pred_proba = [], []
        
        # Accumulate true labels and predicted probabilities
        for batch_x, batch_y in data_generator:
            preds = model.predict(batch_x)
            y_pred_proba.extend(np.max(preds, axis=1))
            y_true.extend(np.argmax(batch_y, axis=1))

        precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba, pos_label=np.argmax(y_true))
        
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, label=f"mAP: {self.metrics['mean_avg_precision']:.2f}")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, normalize=False):
        # Plot confusion matrix as a heatmap
        if self.confusion_matrix is None:
            print("No confusion matrix to display. Run calculate_metrics first.")
            return

        cm = self.confusion_matrix
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix on Test Set")
        plt.show()


if __name__ == '__main__':
    project_dir=""
    test_dir = project_dir + 'Data Set Ordered/test data/'

    batch_size = 32

    test_generator = HomemadeDataloader(test_dir, batch_size, (224,224), isValidation=True, preprocess_function=preprocess_input)
    # Instantiate the MatrixPlotter class
    plotter = MatrixPlotter()

    # Load the model and data generator
    model = keras.models.load_model('models/best_model_iteration_4.h5')

    # Calculate metrics and plot them
    plotter.calculate_metrics(model, test_generator)
    plotter.print_metrics()
    plotter.plot_confusion_matrix(normalize=True)
    plotter.plot_precision_recall_curve(test_generator)