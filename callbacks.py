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


class ZeroShotKNNTestCallback(Callback):

    def __init__(self, cb_freq: int = 1, log_fname: str = None):
        raise Exception("TODO: update with new dataloading")
        self.knn_train = create_img_label_ds(split="knn/train", img_hw=640).batch(16)
        self.knn_test = create_img_label_ds(split="knn/test", img_hw=640).batch(16)
        self.log = None if log_fname is None else open(log_fname, "w")
        self.cb_freq = cb_freq

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.cb_freq == 0:
            x_train, y_train = generate_embeddings_from_model(
                self.model, self.knn_train
            )
            x_test, y_test = generate_embeddings_from_model(self.model, self.knn_test)
            report = check(x_train, y_train, x_test, y_test)
            for k in ["accuracy", "macro avg", "weighted avg"]:
                del report[k]
            logs["mean_f1"] = (
                report["cat"]["f1-score"] + report["dog"]["f1-score"]
            ) / 2
            report["epoch"] = epoch
            if self.log:
                print(json.dumps(report), file=self.log, flush=True)
        return logs
