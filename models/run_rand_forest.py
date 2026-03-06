import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd 
import numpy as np
import yaml

from models.random_forest import RandForestPredictor
from helper import metric_heatmap

import pickle
from joblib import Parallel, delayed

import time
from datetime import datetime


def run_dataset(data, output_folder, params):
    print(data)

    # Initialize predictor
    rf_predictor = RandForestPredictor(data_name=data, output_folder=output_folder) 
    #rf_predictor = RandForestPredictor(data_name=data, dataset_name='kaggle', sequence_column="junction_aa", sample_column="sample", label_column="label_positive", output_folder='output_stats/kaggle')
                                    
    # Run nested CV for evaluation and fit final model
    rf_predictor.nested_cv(
        params=params,
        n_iter=20,
        n_splits=5,
        shuffle=True
    )
    rf_predictor.get_consensus_params()
    
    rf_predictor.fit_final_model(n_iter=20)

    #rf_predictor.confusion_matrix(filename=f"confusion_matrix_{data}.png")
    #rf_predictor.explore_decision_trees(filename=f"trees_{data}/tree")
    #rf_predictor.feature_importance(filename=f"feature_importance_{data}.png")
    return data, rf_predictor



if __name__ == '__main__':
    
    # simulated datasets
    with open('lib/airr_datasets.yaml', 'r') as f:
        airr_datasets = yaml.load(f, Loader=yaml.SafeLoader)

    # airr_datasets 
    ## old simulated datasets
    # first 10 datasets: motif with 3 AA --> control: AAA, disease: YYG
    # next 10 datasets: motif with 13 AA --> control: HNDYSEIRCVLQN, disease: GPKALMVFWQRST
    ## new simulated datasets
    # all other 270 datasets

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

    datasets_groups = datasets_df.groupby(['size', 'seed']).apply(lambda g: g.index.tolist(), include_groups=False).to_dict() # 15 groups with 18 datasets each


    #datasets_groups = {(50, 3): ['simulated_seed3_freq010_size50_noise5_dataset', 'simulated_seed3_freq010_size50_noise25_dataset'],
    #                   (50, 5): ['simulated_seed5_freq010_size50_noise5_dataset', 'simulated_seed5_freq010_size50_noise25_dataset',]}
    
    # kaggle datasets
    #datasets_kaggle = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5', 'dataset_6', 'dataset_7_1', 'dataset_7_2', 'dataset_8_1', 'dataset_8_2', 'dataset_8_3', 'dataset_7', 'dataset_8']
    
    
    
    # params for HP tuning
    params = {
        'model__max_features': ['sqrt'], # number of features to consider when looking for the best split 
        # higher: few relevant variables, high dimensional data
        # lower: many relevant variables, more diverse trees, on avg worse performance, reduces runtime
        'model__max_samples': [None, 0.8], # If bootstrap is True, the number of samples to draw from X to train each base estimator.
        # higher:
        # lower: more diverse trees, single trees worse performance, for most datasets better performance, reduces runtime
        'model__bootstrap': [True], # True: replacement, False: without replacement
        'model__min_samples_leaf': [1, 5, 10], # minimum number of samples required to be at a leaf node
        # higher: more noise variables, reduces runtime, large datasets
        # lower: trees with larger depth
        'model__n_estimators': [100, 200, 500, 1000], # number of trees in the forest
        # higher: better, low sample size & high node size & small mtry, increases runtime
        # lower: 
        'model__criterion': ['gini'], # function to measure the quality of the split: {“gini”, “entropy”, “log_loss”}
        'model__class_weight': ["balanced", "balanced_subsample"]
    }


    output_folder = "output_stats/sim_270_2"

    DATASET_PARALLEL = int(max(1, os.cpu_count() // 4)) # max(1, int(3/4 * os.cpu_count()))
    
    for group in list(datasets_groups.keys())[:]:
        start = time.time()

        print()
        print(f'size: {group[0]}, seed: {group[1]}')
        print("Start:", datetime.fromtimestamp(start).strftime("%H:%M:%S"))

        rf_models = {}

        results = Parallel(n_jobs=DATASET_PARALLEL)(
            delayed(run_dataset)(data, output_folder, params)
            for data in datasets_groups[group]
        )
        rf_models = dict(results)

        # Save to file
        with open(f'{output_folder}/rf_variables_size{group[0]}_seed{group[1]}.pkl', 'wb') as f:
            pickle.dump(rf_models, f)

        metric_heatmap(rf_models, filename=f'{output_folder}/metrics_heatmap_size{group[0]}_seed{group[1]}.png')

        end = time.time()
        print("End:", datetime.fromtimestamp(end).strftime("%H:%M:%S"))
        print("Duration:", end - start, "seconds")
        print()



    
