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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_predict, cross_validate
from scipy.stats import randint
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

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
        self.hp_params = None
        self.best_params = None
        self.nested_scores = None
        self.y_pred = None

        #tuning params
        self.opt_metric = 'average_precision'

        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
    

    def make_pipeline(self):
        return Pipeline([
            ('sampler', RandomUnderSampler()),    # balance the classes
            ("scaler", StandardScaler()),
            ("model", self.model)
        ])

    def nested_cv(self, params, n_iter=10, n_splits=5, shuffle=True):
        self.hp_params = params
        pipe = self.make_pipeline()
        
        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self.hp_params,
            n_iter=n_iter,
            cv=inner_cv,
            scoring=self.opt_metric,
            n_jobs=-1,
            random_state=self.random_state
        )

        # Outer CV for unbiased evaluation
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)

        scores = cross_validate(
            search,
            self.X_train,
            self.y_train,
            cv=outer_cv,
            scoring=["accuracy", "balanced_accuracy", "precision", "recall", "roc_auc", "average_precision"],
            return_estimator=True,
            n_jobs=-1
        )

        self.nested_scores = scores

        for metric in ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 'test_recall', 'test_roc_auc', 'test_average_precision']:
            print("Nested CV {}: {:.3f} ± {:.3f}".format(metric.removeprefix("test_"),
                scores[metric].mean(), scores[metric].std()))
        

        return scores


    def fit_final_model(self, n_iter=10, n_splits=5):
        """
        Trains a final model on the full dataset using RandomizedSearchCV.
        """
        pipe = self.make_pipeline()
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self.hp_params,
            n_iter=n_iter,
            cv=n_splits,
            scoring=self.opt_metric,
            n_jobs=-1,
            random_state=self.random_state
        )
        search.fit(self.X_train, self.y_train)
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        print("Final model trained with hyperparameters:", self.best_params)
        print()

        self.predict()


    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    
    def confusion_matrix(self, filename="confusion_matrix.png"):
        if self.y_pred is None:
            self.predict()
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/{filename}')
        plt.close()
    

    def explore_decision_trees(self, n=3, max_depth=2, filename='trees/tree'):
        rf = self.model.named_steps['model']
        for i in range(min(n, len(rf.estimators_))):
            
            tree = rf.estimators_[i]
            dot_data = export_graphviz(tree,
                                    feature_names=self.X_train.columns,  
                                    filled=True,  
                                    max_depth=max_depth, 
                                    impurity=False, 
                                    proportion=True)
            graph = graphviz.Source(dot_data)
            graph.render(f"{self.output_folder}/{filename}_{i}", format="pdf", cleanup=True)
    

    def feature_importance(self, n=10, filename="feature_importances.png"):
        importances = pd.Series(self.model.named_steps['model'].feature_importances_, index=self.X_train.columns)
        ax = importances.sort_values(ascending=False).head(n).plot.bar(figsize=(10, 6))
        ax.set_title(f"Top {n} Feature Importances")

        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/{filename}')
        plt.close()



if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    # control: AAA, disease: YYG
    datasets_3AA = ['simulated_200_balanced_dataset', 'simulated_200_unbalanced_dataset', 'simulated_500_balanced_dataset', 'simulated_500_unbalanced_dataset', 'simulated_1k_balanced_dataset', 'simulated_1k_unbalanced_dataset', 'simulated_2k_balanced_dataset', 'simulated_2k_unbalanced_dataset', 'simulated_2k_balanced_noisy_05_dataset', 'simulated_2k_balanced_noisy_25_dataset']
    # control: HNDYSEIRCVLQN, disease: GPKALMVFWQRST
    datasets_13AA = [str.replace(data,'dataset', '13AA_dataset') for data in datasets_3AA]
    
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

    rf_models = {}
    for data in datasets_13AA:
        print(data)

        # ------------------ 2. Initialize predictor ------------------
        rf_predictor = RandForestPredictor(data_name=data, output_folder='output_stats/13AA')

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
        rf_predictor.explore_decision_trees(filename=f"trees_{data}/tree")
        rf_predictor.feature_importance(filename=f"feature_importance_{data}.png")

        rf_models[data] = rf_predictor



    # Save to file
    with open('output_stats/13AA/random_forest_variables.pkl', 'wb') as f:
        pickle.dump(rf_models, f)

    metric_heatmap(rf_models, filename='output_stats/13AA/metrics_heatmap.png')

    """
    # Load later
    with open('random_forest_variables.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data.keys())
    print(data['simulated_200_balanced_dataset'].label_column)

    """
