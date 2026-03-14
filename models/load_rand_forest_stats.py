import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import pickle

from models.helper import preprocess_data, metric_heatmap
from models.random_forest import RandForestPredictor
import lib.metrics as me


# extract the properties of each dataset from its descriptive name and save them as columns in the dataframe
def get_dataset_properties(df, col, sim):
    if sim:
        df[["simulated", "seed", "freq", "size", "noise", "data"]] = df[col].str.split('_', expand=True)
        df = df.drop(columns=["simulated", "data"])

        for col in ["seed", "freq", "size", "noise"]:
            df[col] = df[col].str.replace(col, "").astype(int)
    else:
        df[["data", "number", "random_seed"]] = df[col].str.split('_', expand=True)
        df["dataset"] = df[["data", "number"]].agg("_".join, axis=1)
        df = df.drop(columns=["data", "number"])
        df["random_seed"] = df["random_seed"].astype(int)
    return df


if __name__ == '__main__':
    # parameters to change
    folder_path = "/vol/data/immuneML/output_rf/sim_270/"    # output folder from run_rand_forest.py script
    output_folder = 'plots/stats/sim_270'                    # output folder for results from this script
    sim = True     # set True for simulated datasets and False for kaggle datasets

    os.makedirs(output_folder, exist_ok=True)

    motif_dfs = []  # list of dataframes containing the most important features for all final models
    metrics_best_models = {}    # dict with dataset names as keys and the calculated metrics for all final models as values 
    nested_cv_hyps_dict = {}    # dict with dataset names as keys and the hyperparameters from the nested CV folds as values (all models)
    nested_scores = []  # list of dicts containing the metric values for the nested CV folds (all models)
    used_hyps = {}  # dict with dataset names as keys and the hyperparameters for all final models as values

    # iterate over .pkl files in folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "rb") as f:
                my_data = pickle.load(f)
            
            
            rf_dict = {}    # dict with dataset names as keys and Series of the most important features for each final model as values
            # iterate over the final models for each dataset in the file
            for key, rf in my_data.items():
                # get the 10 most important features for the final model 
                importances = pd.Series(rf.best_model.named_steps['model'].feature_importances_, index=rf.X_train.columns)
                ax = importances.sort_values(ascending=False).head(10)
                rf_dict[key] = ax

                # calculate the chosen metrics for the final model
                metrics = me.calc_metrics(rf.y_test, rf.best_model.predict_proba(rf.X_test)[:, 1])
                metrics_best_models[key] = metrics

                # get hyperparameters for nested cv folds
                nested_cv_hyps_dict[key] = rf.get_consensus_params()

                # get metrics for nested cv folds
                scores = rf.nested_scores
                n_folds = len(scores["test_accuracy"])

                for fold in range(n_folds):
                    row = {"dataset_name": key, "fold": fold}
                    for metric, values in scores.items():
                        if isinstance(values, (list, tuple)) or hasattr(values, "__len__"):
                            if metric != "estimator":  # skip estimator objects
                                row[metric.removeprefix("test_")] = values[fold]
                    nested_scores.append(row)

                # used hyperparameters
                used_hyps[key] = rf.best_params

            # convert most important feature dict into dataframe
            motif_col = pd.DataFrame(rf_dict)
            motif_col = motif_col.transpose()
            
            motif_dfs.append(motif_col)



    # concatenate the list of dataframes containing the most important features for all final models to a dataframe
    full_df = pd.concat(motif_dfs)
    full_df = full_df.reset_index(names='dataset_name')
    full_df = get_dataset_properties(full_df, 'dataset_name', sim)
    full_df.to_csv(f'{output_folder}/in_top10_features.csv', index=False)



    # concatenate the dicts with the hyperparameters for the nested CV folds to a dataframe
    nested_cv_hyps_df = pd.concat([
        df.dropna(axis=1, how='all').assign(key=k)  # drop empty/all-NA columns
        for k, df in nested_cv_hyps_dict.items()],
        ignore_index=False)
    nested_cv_hyps_df = (nested_cv_hyps_df
                         .rename(columns=lambda c: c.replace("model__", ""))  # remove prefix
                         .rename(columns={"key": "dataset_name"})                  # rename column
                         .reset_index(names="fold")                           # index -> column
                         )
    # convert the list of dicts containing the metric values for the nested CV folds to a dataframe
    nested_cv_scores_df = pd.DataFrame(nested_scores)
    
    # combine the dataframes for the hyperparameters and metric values for the nested CV folds
    combined_df = pd.merge(nested_cv_scores_df, nested_cv_hyps_df, on=["dataset_name", "fold"], how="inner")
    combined_df = get_dataset_properties(combined_df, 'dataset_name', sim)
    combined_df.to_csv(f'{output_folder}/nested_cv_stats.csv', index=False)



    # convert the dicts with the metric values for the final models to a dataframe
    df_metrics = pd.DataFrame.from_dict(metrics_best_models, orient='index')
    df_metrics = df_metrics.reset_index(names='dataset_name')
    
    # convert the dicts with the hyperparameters for the final models to a dataframe
    used_hyps_df = pd.DataFrame.from_dict(used_hyps, orient='index')
    used_hyps_df = (used_hyps_df
                         .rename(columns=lambda c: c.replace("model__", ""))  # remove prefix
                         .reset_index(names="dataset_name")                        # index -> column
                         )
    
    # combine the dataframes for the hyperparameters and metric values for the final models
    comb_df = pd.merge(df_metrics, used_hyps_df, on='dataset_name', how='inner')
    comb_df = get_dataset_properties(comb_df, 'dataset_name', sim)
    comb_df.to_csv(f'{output_folder}/model_stats.csv', index=False)


    
