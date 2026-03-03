import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.random_forest import RandForestPredictor
from helper import metric_heatmap

import pickle



if __name__ == '__main__':
    # old simulated datasets

    # control: AAA, disease: YYG
    datasets_3AA = ['simulated_200_balanced_dataset', 'simulated_200_unbalanced_dataset', 'simulated_500_balanced_dataset', 'simulated_500_unbalanced_dataset', 'simulated_1k_balanced_dataset', 'simulated_1k_unbalanced_dataset', 'simulated_2k_balanced_dataset', 'simulated_2k_unbalanced_dataset', 'simulated_2k_balanced_noisy_05_dataset', 'simulated_2k_balanced_noisy_25_dataset']
    # control: HNDYSEIRCVLQN, disease: GPKALMVFWQRST
    datasets_13AA = [str.replace(data,'dataset', '13AA_dataset') for data in datasets_3AA]

    # kaggle datasets
    datasets_kaggle = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5', 'dataset_6', 'dataset_7_1', 'dataset_7_2', 'dataset_8_1', 'dataset_8_2', 'dataset_8_3', 'dataset_7', 'dataset_8']
    
    # new simulated datasets
    datasets_test = ['simulated_1000_balanced_test_no_noisy_13_AA', 'simulated_1000_balanced_test_noisy_13_AA']

    datasets_sim = ['variant_seed3_freq025_size150_noise70', 'variant_seed3_freq025_size200_noise80', 'variant_seed3_freq025_size50_noise25', 'variant_seed3_freq025_size50_noise5', 'variant_seed3_freq025_size50_noise70', 'variant_seed3_freq025_size50_noise80', 'variant_seed3_freq050_size200_noise80', 'variant_seed3_freq050_size50_noise25', 'variant_seed5_freq010_size150_noise5', 'variant_seed5_freq010_size150_noise60', 'variant_seed5_freq025_size100_noise80', 'variant_seed5_freq025_size200_noise80', 'variant_seed5_freq050_size100_noise60', 'variant_seed5_freq050_size50_noise50', 'variant_seed7_freq025_size200_noise5', 'variant_seed7_freq025_size200_noise70', 'variant_seed7_freq025_size50_noise5', 'variant_seed7_freq025_size50_noise50', 'variant_seed7_freq050_size200_noise50', 'variant_seed7_freq050_size200_noise70']
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
        'model__class_weight': ["balanced"] #, "balanced_subsample"]
    }


    output_folder = "output_stats/new" # 'output_stats/13AA'


    rf_models = {}
    for data in datasets_sim:
        print(data)

        # ------------------ 2. Initialize predictor ------------------
        rf_predictor = RandForestPredictor(data_name=data, output_folder=output_folder) 
        #rf_predictor = RandForestPredictor(data_name=data, dataset_name='kaggle', sequence_column="junction_aa", sample_column="sample", label_column="label_positive", output_folder='output_stats/kaggle')
                                        
        
        # ------------------ 3. Run nested CV for evaluation ------------------
        
        nested_scores = rf_predictor.nested_cv(
            params=params,
            n_iter=12,
            n_splits=5,
            shuffle=True
        )

        # Nested CV prints mean ± std of average precision and recall

        # ------------------ 4. Train final model on full dataset ------------------
        rf_predictor.fit_final_model(n_iter=20)

        rf_predictor.confusion_matrix(filename=f"confusion_matrix_{data}.png")
        #rf_predictor.explore_decision_trees(filename=f"trees_{data}/tree")
        rf_predictor.feature_importance(filename=f"feature_importance_{data}.png")

        rf_models[data] = rf_predictor



    # Save to file
    with open(f'{output_folder}/random_forest_variables.pkl', 'wb') as f:
        pickle.dump(rf_models, f)

    metric_heatmap(rf_models, filename=f'{output_folder}/metrics_heatmap.png')
