from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import sys
import json

from util import ensure_dir_exists_for_file

class EvalCallback(Callback):

    def __init__(
        self, names: str, datasets, class_names: str, report_dir: str, cb_freq: int = 1
    ):
        assert len(names) == len(datasets)
        self.names = names
        self.datasets = datasets
        self.class_names = class_names
        self.report_dir = report_dir
        self.cb_freq = cb_freq

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.cb_freq != 0:
            return

        eval_report = {"epoch": epoch}
        for name, dataset in zip(self.names, self.datasets):
            # calc argmax classifications
            y_trues, y_preds = [], []
            for x, y_true in dataset:
                y_trues.append(np.array(y_true).flatten())
                y_preds.append(np.argmax(self.model(x), axis=-1))

            # calc classification report and add to eval_report
            report = classification_report(
                y_true=np.concatenate(y_trues),
                y_pred=np.concatenate(y_preds),
                target_names=self.class_names,
                output_dict=True,
            )
            eval_report[name] = report

            # add the class -> f1-scores, as well as the harmonic mean to the logs
            reciprocal_sum = 0
            for class_name in self.class_names:
                class_f1 = report[class_name]["f1-score"]
                logs[f"{class_name}_f1/{name}"] = class_f1
                reciprocal_sum += 1 / (class_f1 + sys.float_info.epsilon)
            logs[f"harmonic_f1/{name}"] = len(self.class_names) / reciprocal_sum

        fname = f"{self.report_dir}/e{epoch:03d}_report.json"
        ensure_dir_exists_for_file(fname)
        with open(fname, "w") as f:
            json.dump(eval_report, f)

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
