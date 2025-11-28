from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


class EvalCallback(Callback):

    def __init__(self, name: str, dataset, class_names: str, cb_freq: int = 1):
        self.name = name
        self.dataset = dataset
        self.class_names = class_names
        self.cb_freq = cb_freq

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.cb_freq != 0:
            return
        y_trues = []
        y_preds = []
        for x, y_true in self.dataset:
            y_trues.append(np.array(y_true).flatten())
            y_preds.append(np.argmax(self.model(x), axis=-1))
        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)
        confusion = confusion_matrix(y_true=y_trues, y_pred=y_preds)
        report = classification_report(
            y_true=y_trues, y_pred=y_preds, target_names=self.class_names
        )
        print(self.name, "\n", confusion, "\n", report)

        return logs
