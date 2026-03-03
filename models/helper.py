import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import lib.dataloader as dl
import lib.datasplit as ds
import lib.metrics as me
import encoding.kmer_freq as kf


# load dataset, do kmer encoding and split data (stratified)
def preprocess_data(data, dataset_name='airr', k=3, seq_col="cdr3_aa", samp_col="sample", lab_col="disease", test_size=0.2, random_state=42):
    valid_dataset_names = ['airr', 'kaggle']
    if dataset_name not in valid_dataset_names:
        print(f"Choose a valid dataset name. {valid_dataset_names}")
        return
    
    if dataset_name == 'airr':
        df = dl.load_airr_dataset(data)
    elif dataset_name == 'kaggle':
        df = dl.load_kaggle_dataset(data)
    
    df = kf.encode_repertorie_normalized(df, k=k, sequence_column=seq_col, sample_column=samp_col, label_column=lab_col)
    print (f"Encoded {len(df)} samples with k-mer frequencies.")
    
    df = df.set_index(samp_col)

    # Split the data into features (X) and target (y)
    X = df.drop(columns=lab_col, axis=1)
    y = df[lab_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


# calculate metrics for a dict of ml models and save as heatmap (png)
## ml_dict: {name of dataset: class with parameters (y, y_pred), ...}
def metric_heatmap(ml_dict, filename="metrics_heatmap.png"):

    metrics_best_models = {}

    for data in ml_dict.keys():

        rf_model = ml_dict[data]
        metrics = me.calc_metrics(rf_model.y_test, rf_model.y_pred)
        metrics_best_models[data] = metrics

    df_metrics = pd.DataFrame.from_dict(metrics_best_models, orient='index')

   
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(df_metrics.values, aspect="auto", vmin=0, vmax=1)

    for i in range(df_metrics.shape[0]):
        for j in range(df_metrics.shape[1]):
            ax.text(
                j, i, f"{df_metrics.values[i, j]:.2f}",
                ha="center", va="center", color="black"
            )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    # Ticks and labels
    ax.set_xticks(np.arange(df_metrics.shape[1]))
    ax.set_yticks(np.arange(df_metrics.shape[0]))
    ax.set_xticklabels(df_metrics.columns)
    ax.set_yticklabels(df_metrics.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()