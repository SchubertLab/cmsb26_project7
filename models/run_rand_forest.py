import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd 
import numpy as np
import yaml

import pickle
from joblib import Parallel, delayed

import time
from datetime import datetime


from models.random_forest import RandForestPredictor
from helper import metric_heatmap


# initialize and run my Random Forest workflow
def run_dataset(data, output_folder, params, sim, n_jobs=1, random_state=42):
    # Initialize predictor 
    if sim:
        # for simulated datasets
        rf_predictor = RandForestPredictor(data_name=data, n_jobs=n_jobs, output_folder=output_folder)
    else:
        # for kaggle datasets
        rf_predictor = RandForestPredictor(data_name=data, dataset_name='kaggle', sequence_column="junction_aa", sample_column="sample", label_column="label_positive", random_state=random_state, n_jobs=n_jobs, output_folder=output_folder) 
                                    
    # Run nested CV for evaluation and fit final model
    rf_predictor.nested_cv(
        params=params,
        n_iter=30,
        n_splits=5,
        shuffle=True
    )
    rf_predictor.get_consensus_params()
    
    rf_predictor.fit_final_model(n_iter=30)

    return data, rf_predictor



if __name__ == '__main__':
    # parameters to change
    output_folder = "/vol/data/immuneML/output_rf/sim_270" # "/vol/data/immuneML/output_rf/kaggle"
    sim = True         # set True for simulated datasets and False for kaggle datasets


    # params for HP tuning
    params = {
        #'model__max_features': ['sqrt'], # number of features to consider when looking for the best split 
        # higher: few relevant variables, high dimensional data
        # lower: many relevant variables, more diverse trees, on avg worse performance, reduces runtime
        'model__max_samples': [None, 0.8], # If bootstrap is True, the number of samples to draw from X to train each base estimator.
        # higher:
        # lower: more diverse trees, single trees worse performance, for most datasets better performance, reduces runtime
        #'model__bootstrap': [True], # True: replacement, False: without replacement
        'model__min_samples_leaf': [1, 5, 10], # minimum number of samples required to be at a leaf node
        # higher: more noise variables, reduces runtime, large datasets
        # lower: trees with larger depth
        'model__n_estimators': [100, 200, 500], # 1000 # number of trees in the forest
        # higher: better, low sample size & high node size & small mtry, increases runtime
        # lower: 
        #'model__criterion': ['gini'], # function to measure the quality of the split: {“gini”, “entropy”, “log_loss”}
        #'model__class_weight': ["balanced", "balanced_subsample"],
        'model__min_samples_split': [2, 5, 10],
        'model__max_depth': [None, 10, 20]
    }

    if sim:
        # load descriptive names of the simulated datasets
        with open('lib/airr_datasets.yaml', 'r') as f:
            airr_datasets = yaml.load(f, Loader=yaml.SafeLoader)

        # airr_datasets 
        ## old simulated datasets
        # first 10 datasets: motif with 3 AA --> control: AAA, disease: YYG
        # next 10 datasets: motif with 13 AA --> control: HNDYSEIRCVLQN, disease: GPKALMVFWQRST
        ## new simulated datasets
        # all other 270 datasets

        # filter for new simulated datasets
        datasets = [key for key in airr_datasets.keys() if key.startswith('simulated_seed')]
        
        # get dataset properties from its name and save as dataframe
        datasets_df = pd.DataFrame(
            [s.split("_") for s in datasets],
            index=datasets,
            columns=["simulated", "seed", "freq", "size", "noise", "dataset"]
        )
        datasets_df = datasets_df.drop(columns=["simulated", "dataset"])
        for col in datasets_df.columns:
            datasets_df[col] = datasets_df[col].str.replace(col, "").astype(int)

        # group simulated datasets by their size and seed length --> 15 groups with 18 datasets each
        datasets_groups = datasets_df.groupby(['size', 'seed']).apply(lambda g: g.index.tolist(), include_groups=False).to_dict() 
    else:
        # kaggle datasets in 
        datasets_groups = {key: ['dataset_7', 'dataset_8'] for key in [42, 43, 44, 45, 46]} # different random states

    n_jobs = 5 if sim else 15
    parallel_datasets = int(max(1, os.cpu_count() // n_jobs )) 

    # iterate over groups, execute Random Forest workflow for each dataset in the group and save results
    for group in list(datasets_groups.keys()):
        print()
        start = time.time()
        print(f"size: {group[0]}, seed: {group[1]}" if sim else f"random state: {group}")
        print("Start:", datetime.fromtimestamp(start).strftime("%H:%M:%S"))

        # execute Random Forest workflow for the datasets in the group                                                                               
        rf_models = {}
        results = Parallel(n_jobs=parallel_datasets)(
            delayed(run_dataset)(data, output_folder, params, sim, n_jobs=n_jobs, random_state=42 if sim else group)
            for data in datasets_groups[group]
        )

        if sim:
            rf_models = dict(results)
        else:
            rf_models.update({f"{key}_{group}": value for key, value in results})

        # Save the Random Forest models dict as pkl file
        filename_addition = f'size{group[0]}_seed{group[1]}' if sim else f'random_state_{group}'

        with open(f'{output_folder}/rf_variables_{filename_addition}.pkl', 'wb') as f:
            pickle.dump(rf_models, f)

        # visualize the metrics for the final Random Forest models as heatmap
        metric_heatmap(rf_models, filename=f'{output_folder}/metrics_heatmap_{filename_addition}.png')

        end = time.time()
        print("End:", datetime.fromtimestamp(end).strftime("%H:%M:%S"))
        print("Duration:", end - start, "seconds")
        print()



    
