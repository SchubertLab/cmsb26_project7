import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.metrics as me
from models.helper import preprocess_data
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold, cross_validate

class LogRegPredictor:
    def __init__(self, data, dataset_name='airr', seq_col='cdr3_aa', sample_column='sample', label_column='disease', random_state=16, split_seed=42):
        #data
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(
            data=data,
            dataset_name=dataset_name,
            seq_col=seq_col,
            samp_col=sample_column, 
            lab_col=label_column,
            random_state=split_seed)
        
        #model
        self.model = None
        self.best_params = None
        self.random_state = random_state
        self.y_prob = None
        self.metrics = None #me.calc_metrics()

        #model params
        self.penalty = 'elasticnet'
        self.solver = 'saga'
        self.max_iter = 5000
        self.l1_ratio = 0.5
        self.C = 1

              #tuning params
        self.opt_metric = 'average_precision'
        self.c_values = np.logspace(-4,2,15)
        self.l1_ratio_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        self.tuning_parameters = {
            'model__C' : self.c_values,
            'model__l1_ratio': self.l1_ratio_grid}
        #randomized search
        self.n_iter = 40

    def make_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                penalty=self.penalty,
                solver=self.solver,
                C=self.C,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
                max_iter=self.max_iter,
                class_weight='balanced'
            ))
        ])

    def _tune_model(self):
        pipe = self.make_pipeline()
    
        random_search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self.tuning_parameters,
            scoring= self.opt_metric,
            n_iter=self.n_iter,
            n_jobs=-1,
            cv=5,
            random_state=self.random_state
        )

        random_search.fit(self.X_train, self.y_train)

        return random_search.best_estimator_, random_search.best_params_

    def nested_cv(self, n_splits=5):
        pipe = self.make_pipeline()

        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self.tuning_parameters,
            n_iter=self.n_iter,
            cv=inner_cv,
            scoring=self.opt_metric,
            n_jobs=-1,
            random_state=self.random_state
        )

        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        scores = cross_validate(
            search,
            self.X_train,
            self.y_train,
            cv=outer_cv,
            scoring=['accuracy', 'balanced_accuracy', 'precision', 'recall', 'roc_auc', 'average_precision'],
            return_estimator=True,
            n_jobs=-1
        )

        self.nested_scores = scores

        for metric in ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 'test_recall', 'test_roc_auc', 'test_average_precision']:
            print("Nested CV {}: {:.3f} ± {:.3f}".format(
                metric.removeprefix("test_"),
                scores[metric].mean(),
                scores[metric].std()
            ))

        return scores

    def train(self, tune=True):
        if tune:
            print(f"Tuning and Training LogisticRegression on {self.X_train.shape[0]} samples and {self.X_train.shape[1]} kmer features")
            self.model, self.best_params = self._tune_model()
            print("Tuning and training complete.")
            print(f"Best model parameters: {self.best_params}")
        else:
            print(f"Training LogisticRegression on {self.X_train.shape[0]} samples and {self.X_train.shape[1]} kmer features")
            self.model = self.make_pipeline()
            self.model.fit(self.X_train, self.y_train)
            print("Training complete.")

    def predict(self):
        print(f"Testing LogisticRegression on {self.X_test.shape[0]} samples")
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
        self.y_pred = (self.y_prob >= 0.5).astype(int)
        print("Testing complete.")
        return self.y_prob

    def evaluate(self):
        if self.y_prob is None:
            raise ValueError("No predicted probabilities found. Run predict() first.")
        
        disease_int = self.y_test.astype(int)
        self.metrics = me.calc_metrics(disease_int, self.y_prob)
        print(classification_report(self.y_test, self.y_pred))
        return self.metrics

if __name__ == "__main__":
    model = LogRegPredictor("simulated_200_balanced_dataset",sample_column='sample',label_column='disease')
    model.train()
    model.predict()
    metrics = model.evaluate()
    print(metrics)