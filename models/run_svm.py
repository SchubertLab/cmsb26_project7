import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd 
import numpy as np
import yaml

from models.svm import SVMPredictor
from helper import metric_heatmap

import pickle
from joblib import Parallel, delayed

import time
from datetime import datetime


def run_dataset(data, output_folder, params):
    print(data)

    # Initialize predictor
    # for kaggle data
    svm_predictor = SVMPredictor(data_name=data, dataset_name='kaggle', sequence_column="junction_aa", sample_column="sample", label_column="label_positive", random_state=42,output_folder=output_folder)
    # for simulated data
    #svm_predictor = SVMPredictor(data_name=data, sample_column="sample", output_folder=output_folder)


    # Run nested CV for evaluation and fit final model
    svm_predictor.nested_cv(
        params=params,
        n_iter=30,
        n_splits=5,
        shuffle=True
    )
    
    svm_predictor.fit_final_model(n_iter=30)
    
    # Save individual model
    model_filename = os.path.join(output_folder, f"svm_model_{data}.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(svm_predictor, f)
    print(f"Saved model to {model_filename}")

    return data, svm_predictor



if __name__ == '__main__':
    
    # simulated datasets
    with open('lib/airr_datasets.yaml', 'r') as f:
        airr_datasets = yaml.load(f, Loader=yaml.SafeLoader)

    # all 270 final simulated datasets
    datasets = [key for key in airr_datasets.keys() if key.startswith('simulated_seed')]
    
    # portion datasets
    datasets_df = pd.DataFrame(
        [s.split("_") for s in datasets],
        index=datasets,
        columns=["simulated", "seed", "freq", "size", "noise", "dataset"]
    )
    datasets_df = datasets_df.drop(columns=["simulated", "dataset"])
    for col in datasets_df.columns:
        datasets_df[col] = datasets_df[col].str.replace(col, "").astype(int)

    
    # kaggle datasets
    #datasets_groups = {'kaggle': ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5', 'dataset_6', 'dataset_7_1', 'dataset_7_2', 'dataset_8_1', 'dataset_8_2', 'dataset_8_3', 'dataset_7', 'dataset_8']}
    
    datasets_groups = {'kaggle': ['dataset_7', 'dataset_8']}

    
    
    # params for HP tuning - SVM specific parameters
    params = {
        'model__C': [0.1, 1, 10, 100, 1000],  # Regularization parameter
        # Higher C: fits training data more closely (risk of overfitting)
        # Lower C: more regularization (may underfit)
        
        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient for RBF, poly, sigmoid
        # gamma='scale': 1 / (n_features * X.var())
        # gamma='auto': 1 / n_features
        # Higher gamma: each training example is influential only close to it (complex decision boundary)
        # Lower gamma: far training examples influence (smoother decision boundary)
        
        'model__kernel': ['rbf', 'linear'],  # Two kernel types to test
        # rbf: Radial Basis Function - non-linear, can handle non-linearly separable data
        # linear: Linear kernel - suitable for linearly separable data, faster
    }


    #output_folder = "/vol/data/immuneML/output_svm/simulated_datasets"
    output_folder = "/vol/data/immuneML/output_svm/kaggle"
    os.makedirs(output_folder, exist_ok=True)

    DATASET_PARALLEL = int(max(1, os.cpu_count() // 5 -2)) # max(1, int(3/4 * os.cpu_count()))
    
    for group in list(datasets_groups.keys())[:]:
        start = time.time()

        print()
        #print(f'size: {group[0]}, seed: {group[1]}')
        print(f'kaggle datasets')
        print("Start:", datetime.fromtimestamp(start).strftime("%H:%M:%S"))

        svm_models = {}

        results = Parallel(n_jobs=DATASET_PARALLEL)(
            delayed(run_dataset)(data, output_folder, params)
            for data in datasets_groups[group]
        )
        svm_models = dict(results)

        # Save to file
        #with open(f'{output_folder}/svm_variables_size{group[0]}_seed{group[1]}.pkl', 'wb') as f:
        #with open(f'{output_folder}/svm_variables_simulated.pkl', 'wb') as f:
        with open(f'{output_folder}/svm_variables_kaggle_46.pkl', 'wb') as f:
            pickle.dump(svm_models, f)

        metric_heatmap(svm_models, filename=f'{output_folder}/metrics_heatmap_kaggle_46.png')


        end = time.time()
        print("End:", datetime.fromtimestamp(end).strftime("%H:%M:%S"))
        print("Duration:", end - start, "seconds")
        print()
