import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import preprocess_data, metric_heatmap

# Data Processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, average_precision_score, recall_score, precision_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, cross_val_predict
from scipy.stats import randint
from sklearn.calibration import CalibratedClassifierCV

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

import pickle




class RandForestPredictor:

    def __init__(self, data_name, sample_column='sample', label_column='disease', sequence_column = 'cdr3_aa', k = 3, random_state=42, output_folder='output_stats'):
        #data
        self.data_name = data_name
        self.sample_column = sample_column
        self.label_column = label_column
        self.sequence_column = sequence_column

        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(data_name, k=k, seq_col=self.sequence_column, samp_col=self.sample_column, lab_col=self.label_column)
        
        #model
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        self.best_params = None
        self.y_pred = None

        #tuning params
        self.opt_metric = 'average_precision'

        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
    

    def hp_tuning(self, params):
        gs = GridSearchCV(self.model, param_grid=params, cv=5, scoring=self.opt_metric, n_jobs=-1)
        # Fit the GridSearchCV on your training data
        gs.fit(self.X_train, self.y_train) 
        self.model = gs.best_estimator_

        # Print the best hyperparameters
        self.best_params = gs.best_params_
        print('Best hyperparameters:',  gs.best_params_)

        self.predict()


    def predict(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    
    def confusion_matrix(self, filename="confusion_matrix.png"):
        if self.y_pred is None:
            self.predict()
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/{filename}')
        plt.close()
    

    def explore_decision_trees(self, n=3, max_depth=2, filename='trees/tree'):
        for i in range(n):
            tree = self.model.estimators_[i]
            dot_data = export_graphviz(tree,
                                    feature_names=self.X_train.columns,  
                                    filled=True,  
                                    max_depth=max_depth, 
                                    impurity=False, 
                                    proportion=True)
            graph = graphviz.Source(dot_data)
            graph.render(f"{self.output_folder}/{filename}_{i}", format="pdf", cleanup=True)
    

    def feature_importance(self, n=10, filename="feature_importances.png"):
        importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns)
        ax = importances.sort_values(ascending=False).head(n).plot.bar(figsize=(10, 5))
        ax.set_title("Feature Importances")

        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/{filename}')
        plt.close()



if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    # control: AAA, disease: YYG
    datasets_3AA = ['simulated_200_balanced_dataset', 'simulated_200_unbalanced_dataset', 'simulated_500_balanced_dataset', 'simulated_500_unbalanced_dataset', 'simulated_1k_balanced_dataset', 'simulated_1k_unbalanced_dataset', 'simulated_2k_balanced_dataset', 'simulated_2k_unbalanced_dataset', 'simulated_2k_balanced_noisy_05_dataset', 'simulated_2k_balanced_noisy_25_dataset']
    # control: HNDYSEIRCVLQN, disease: GPKALMVFWQRST
    datasets_13AA = [str.replace(data,'dataset', '13AA_dataset') for data in datasets_3AA][:-1]
    
    # params for HP tuning
    param_dict = {
    'n_estimators': [100, 200, 500], # number of trees
    #
    'criterion': ['gini'], # ['gini', 'log_loss'], # function to measure the quality of the split: {“gini”, “entropy”, “log_loss”}
    'min_samples_leaf': [1], #[1, 5, 10], # minimum number of samples required to be at a leaf node
    # increasing noise variables leads to higher optimal node size; large datasets higher --> decreases runtime
    'max_features': ['sqrt'], #['sqrt', None],  # number of features to consider when looking for the best split 
    # high if only few relevant variables, low otherwise; higher for high dimensional data
    'bootstrap': [True], # True: replacement, False: without replacement
    # 
    #'class_weight': ["balanced"], # ["balanced_subsample"] #?
    'max_samples': [None], #[None, 4/5], # If bootstrap is True, the number of samples to draw from X to train each base estimator.
    # less better performance
    }

    rf_models = {}
    for data in datasets_3AA[:3]:
        print(data)
        rf_model = RandForestPredictor(data, output_folder=f'output_stats/{data}')

        rf_model.hp_tuning(param_dict)
        
        rf_model.confusion_matrix()
        rf_model.explore_decision_trees()
        rf_model.feature_importance()

        rf_models[data] = rf_model



    # Save to file
    with open('output_stats/random_forest_variables.pkl', 'wb') as f:
        pickle.dump(rf_models, f)

    metric_heatmap(rf_models, filename='output_stats/metrics_heatmap.png')
    
    """
    # Load later
    with open('random_forest_variables.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data.keys())
    print(data['simulated_200_balanced_dataset'].label_column)

    """
