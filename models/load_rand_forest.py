import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import preprocess_data, metric_heatmap
from models.random_forest import RandForestPredictor
import lib.metrics as me

import pickle

import pandas as pd

if __name__ == '__main__':
    # Load later
    folder_path = "/vol/data/immuneML/output_rf/sim_270_no_leakage/"

    data = {}

    motif_dfs = []
    metrics_best_models = {}
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

            # top 10 important features
            df = pd.DataFrame(rf_dict)
            motif_col = df.reindex(['GPK', 'PKA', 'KAL', 'ALM', 'LMV'])
            motif_col = motif_col.transpose()
            #motif_col['filename'] = filename
            
            motif_dfs.append(motif_col)


    # top 10 important features
    full_df = pd.concat(motif_dfs)
    full_df.to_csv('in_top10_features_nl.csv')

    # metrics
    df_metrics = pd.DataFrame.from_dict(metrics_best_models, orient='index')
    df_metrics.to_csv('metrics_full_nl.csv')

