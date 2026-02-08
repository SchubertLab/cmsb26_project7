import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.dataloader as dl
import lib.datasplit as ds
import lib.metrics as me
import encoding.kmer_freq as kf
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

def prep_data(dataset, k=3, sequence_column="cdr3_aa", sample_column="sample", label_column="disease"):
    df = dl.load_airr_dataset(dataset)
    df = kf.encode_repertorie_normalized(df, k=k, sequence_column=sequence_column, sample_column=sample_column, label_column=label_column)

    print (f"Encoded {len(df)} samples with k-mer frequencies.")

    train, test = ds.split_data(df)
    return train, test

class LogRegPredictor:
    def __init__(self, sample_column='sample', label_column='disease', random_state=16):
        #data
        self.sample_column = sample_column
        self.label_column = label_column
        
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
        self.C = 1

        #tuning params
        self.opt_metric = 'average_precision'
        self.c_values = np.logspace(-3.5,1,10)
        self.l1_ratio = [0.0, 0.25, 0.5, 0.75, 1]
        self.tuning_parameters = {
            'model__C' : self.c_values,
            'model__l1_ratio': self.l1_ratio}
        #randomized search
        self.n_iter = 25


    #prepare feature & label dataframes
    def prepare_data(self, train_df, test_df):
        self.X_train = train_df.drop(columns=[self.sample_column, self.label_column])
        self.y_train = train_df[self.label_column]

        self.X_test = test_df.drop(columns=[self.sample_column, self.label_column])
        self.y_test = test_df[self.label_column]

    def make_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                penalty=self.penalty,
                solver=self.solver,
                C=self.C,
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
            random_state=self.random_state
        )

        random_search.fit(self.X_train, self.y_train)

        return random_search.best_estimator_, random_search.best_params_
    
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


#train, test = prep_data("simulated_2k_balanced_noisy_25_dataset")
#model = LogRegPredictor('sample','disease')
#model.prepare_data(train, test)
#model.train()
#model.predict()
#metrics = model.evaluate()
#print(metrics)