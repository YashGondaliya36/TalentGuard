import os
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from TalentGuard.config.configuration import ModelEvaluationConfig
import numpy as np
import joblib
from pathlib import Path
from TalentGuard.utils.common import save_json




class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        return accuracy, precision, recall, f1, roc_auc

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]].values.ravel() 

        predicted_qualities = model.predict(test_x)

        
        accuracy, precision, recall, f1, roc_auc = self.eval_metrics(test_y, predicted_qualities)

        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc_score": roc_auc
        }
        save_json(path=Path(self.config.metric_file_name), data=scores)

