import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.helper import preprocess_data, metric_heatmap
from models.random_forest import RandForestPredictor
import lib.metrics as me

import pickle

import pandas as pd

def get_dataset_properties(df, col):
    df[["simulated", "seed", "freq", "size", "noise", "data"]] = df[col].str.split('_', expand=True)

    df = df.drop(columns=["simulated", "data"])

    for col in ["seed", "freq", "size", "noise"]:
        df[col] = df[col].str.replace(col, "").astype(int)

    return df


if __name__ == '__main__':
    # Load later
    folder_path = "/vol/data/immuneML/output_rf/kaggle_noleak_newparams/"
    output_folder = 'plots/stats/kaggle'
    sim = False

    os.makedirs(output_folder, exist_ok=True)


    data = {}

    motif_dfs = []
    metrics_best_models = {}
    nested_cv_hyps_dict = {}
    nested_scores = []
    used_hyps = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            
            data[filename] = file_path

            with open(file_path, "rb") as f:
                my_data = pickle.load(f)
            
            
            rf_dict = {}
            for key, rf in my_data.items():
                # top 10 important features
                importances = pd.Series(rf.best_model.named_steps['model'].feature_importances_, index=rf.X_train.columns)
                ax = importances.sort_values(ascending=False).head(10)
                rf_dict[key] = ax

                # metrics
                metrics = me.calc_metrics(rf.y_test, rf.best_model.predict_proba(rf.X_test)[:, 1])
                metrics_best_models[key] = metrics

                # nested cv hyperparameters
                nested_cv_hyps_dict[key] = rf.get_consensus_params()

                # nested cv scores
                scores = rf.nested_scores
                n_folds = len(scores["test_accuracy"])

                for fold in range(n_folds):
                    row = {"dataset": key, "fold": fold}
                    for metric, values in scores.items():
                        if isinstance(values, (list, tuple)) or hasattr(values, "__len__"):
                            if metric != "estimator":  # skip estimator objects
                                row[metric.removeprefix("test_")] = values[fold]
                    nested_scores.append(row)

                # used hyperparameters
                used_hyps[key] = rf.best_params

            # top 10 important features
            df = pd.DataFrame(rf_dict)
            motif_col = df.reindex(['GPK', 'PKA', 'KAL', 'ALM', 'LMV'])
            motif_col = motif_col.transpose()
            
            motif_dfs.append(motif_col)


    # top 10 important features
    if sim:
        full_df = pd.concat(motif_dfs)
        full_df.to_csv(f'{output_folder}/in_top10_features.csv')


    # nested cv hyperparameters and scores
    nested_cv_hyps_df = pd.concat([
        df.dropna(axis=1, how='all').assign(key=k)  # Drop empty/all-NA columns
        for k, df in nested_cv_hyps_dict.items()],
        ignore_index=False)
    nested_cv_hyps_df = (nested_cv_hyps_df
                         .rename(columns=lambda c: c.replace("model__", ""))  # remove prefix
                         .rename(columns={"key": "dataset"})                  # rename column
                         .reset_index(names="fold")                           # index -> column
                         )

    nested_cv_scores_df = pd.DataFrame(nested_scores)
    
    combined_df = pd.merge(nested_cv_scores_df, nested_cv_hyps_df, on=["dataset", "fold"], how="inner")
    if sim:
        combined_df = get_dataset_properties(combined_df, 'dataset')
    combined_df.to_csv(f'{output_folder}/nested_cv_stats.csv', index=False)


    # final models hyperparameters and metrics
    df_metrics = pd.DataFrame.from_dict(metrics_best_models, orient='index')
    df_metrics = df_metrics.reset_index(names='dataset')
    
    used_hyps_df = pd.DataFrame.from_dict(used_hyps, orient='index')
    used_hyps_df = (used_hyps_df
                         .rename(columns=lambda c: c.replace("model__", ""))  # remove prefix
                         .reset_index(names="dataset")                           # index -> column
                         )
    
    comb_df = pd.merge(df_metrics, used_hyps_df, on='dataset', how='inner')
    if sim:
        comb_df = get_dataset_properties(comb_df, 'dataset')
    comb_df.to_csv(f'{output_folder}/model_stats.csv', index=False)


    
