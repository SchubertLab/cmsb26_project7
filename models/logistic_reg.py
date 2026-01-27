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

#delete after fix
import yaml
from pathlib import Path

#issue:
#df = dl.load_airr_dataset("simulated_2k_balanced_noisy_25_dataset")

#delete after fix
dataset_name = "simulated_2k_balanced_noisy_25_dataset"
with open('/vol/data/immuneML/cmsb26_project7/lib/airr_datasets.yaml', 'r') as f:
    yaml_file = yaml.load(f, Loader=yaml.SafeLoader)

dataset_path = Path(yaml_file[dataset_name]['path'])
metadata = dl.load_metadata(dataset_path)
df = dl.load_repertoires_airr(dataset_path, metadata)


df = kf.encode_repertorie_normalized(df, k=3, sequence_column="cdr3_aa", sample_column="sample", label_column="disease")

print (f"Encoded {len(df)} samples with k-mer frequencies.")

train, test = ds.split_data(df)

print(train)
print(test)

def pipeline(train_df, test_df, sample_column, label_column):
    #modifying
    X_train, y_train, X_test, y_test = prepare_data(train_df, test_df, sample_column, label_column)

    #scaling kmer frequencies and training
    model, scaler = train_model(X_train, y_train)

    #scaling and testing
    y_prob = test_model(model, scaler, X_test)

    #evaluation
    metrics = evaluate_model(y_test, y_prob)

    return y_prob, metrics



#prepare feature & label dataframes
def prepare_data(train_df, test_df, sample_column, label_column):
    X_train = train_df.drop(columns=[sample_column, label_column])
    y_train = train_df[label_column]

    X_test = test_df.drop(columns=[sample_column, label_column])
    y_test = test_df[label_column]

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    print(f"Training LogisticRegression on {X_train.shape[0]} samples and {X_train.shape[1]} kmer features")
    #scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=0.1,
        random_state=16,
        max_iter=1000
    )

    model.fit(X_train_scaled, y_train)
    print("Training complete")

    return model, scaler

def test_model(model, scaler, X_test):
    print(f"Testing LogisticRegression on {X_test.shape[0]} samples")
    #scaling
    X_test_scaled = scaler.transform(X_test)

    #predict state and probabilities
    #y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    print("Testing complete")

    return y_prob

def evaluate_model(y_test, y_prob):
    disease_int = y_test.astype(int).tolist()
    pred_scores = me.calc_metrics(disease_int, y_prob)
    return pred_scores


_, metrics = pipeline(train, test, "sample", "disease")
print(metrics)