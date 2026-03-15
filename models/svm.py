import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import preprocess_data, metric_heatmap

# Data Processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modelling
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, average_precision_score, recall_score, precision_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from collections import Counter

from TimeWrapper import TimeWrapper


class SVMPredictor:

    def __init__(self, data_name, dataset_name='airr', sample_column='sample', label_column='disease', sequence_column = 'cdr3_aa', k = 3, random_state=42, output_folder='output_stats'):
        #data
        self.data_name = data_name
        self.sample_column = sample_column
        self.label_column = label_column
        self.sequence_column = sequence_column

        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(data_name, dataset_name, k=k, seq_col=self.sequence_column, samp_col=self.sample_column, lab_col=self.label_column)
        
        #model
        self.random_state = random_state
        self.model = SVC(kernel='rbf',
                        class_weight='balanced',
                        probability=True,
                        random_state=self.random_state)
        self.hp_params = None
        self.best_params = None
        self.best_model = None
        self.nested_scores = None
        self.y_prob = None
        self.y_pred = None

        #tuning params
        self.opt_metric = 'average_precision'

        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.n_jobs = 5
    
   
    def make_pipeline(self):
        """
        return Pipeline([
            ('sampler', RandomUnderSampler()),    # balance the classes
            ("scaler", StandardScaler()),
            ("model", self.model)])
        """ 
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", self.model)])
         

    @TimeWrapper
    def nested_cv(self, params, n_iter=10, n_splits=5, shuffle=True):
        self.hp_params = params
        pipe = self.make_pipeline()
        
        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self.hp_params,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=self.opt_metric,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        # Outer CV for unbiased evaluation
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)

        scores = cross_validate(
            search,
            self.X_train,
            self.y_train,
            cv=outer_cv,
            scoring=["accuracy", "balanced_accuracy", "precision", "recall", "roc_auc", "average_precision"],
            return_estimator=True,
            n_jobs=self.n_jobs
        )

        self.nested_scores = scores

        for metric in ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 'test_recall', 'test_roc_auc', 'test_average_precision']:
            print("Nested CV {}: {:.3f} ± {:.3f}".format(metric.removeprefix("test_"),
                scores[metric].mean(), scores[metric].std()))
        
        return scores

    @TimeWrapper
    def fit_final_model(self, n_iter=10, n_splits=5):
        """
        Trains a final model on the full dataset using RandomizedSearchCV.
        """
        pipe = self.make_pipeline()
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self.hp_params,
            n_iter=n_iter,
            cv=n_splits,
            scoring=self.opt_metric,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        search.fit(self.X_train, self.y_train)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        print("Final model trained with hyperparameters:", self.best_params)

        self.predict()


    def predict(self):
        self.y_prob = self.best_model.predict_proba(self.X_test)[:, 1]
        self.y_pred = self.best_model.predict(self.X_test)
        

    def feature_importance(self, n=10, filename="feature_importances.png"):
        svm_model = self.best_model.named_steps['model']
        
        # Get coefficients (absolute values for importance)
        if hasattr(svm_model, 'coef_'):
            importances = pd.Series(np.abs(svm_model.coef_[0]), index=self.X_train.columns)
            ax = importances.sort_values(ascending=False).head(n).plot.bar(figsize=(10, 6))
            ax.set_title(f"Top {n} Feature Importances (SVM Coefficients)")
            
            plt.tight_layout()
            plt.savefig(f'{self.output_folder}/{filename}')
            plt.close()
        else:
            print("Feature importance not available for this kernel type (use 'linear' kernel for coefficient-based importance)")