TimeWrapper.py
- helper class printing the runtime of a function


helper.py
- contains three functions:
    - preprocess_data: load dataset, do kmer encoding and split data into training and test set
    - encode: encode the dataset and split it into features and targets
    - metric_heatmap: calculate metrics for a dict of ML models and save as heatmap


random_forest.py
- class that implements a machine learning workflow for repertoire classification using a Random Forest model
- including:
    - data preprocessing
    - nested cross-validation with randomized hyperparameter tuning
    - final model training
    - test set prediction
    - result visualizations


run_rand_forest.py
- script for running the Random Forest workflows for the simulated and kaggle datasets
- change:
    - output_folder: folder to save the results
    - sim: set True for simulated datasets and False for kaggle datasets
- run:
    - conda activate rand_forest_env
    - python models/run_rand_forest.py > log.txt


load_rand_forest_stats.py
- script for saving the most important features as well as the metrics scores and hyperparameters for the nested CV and the final models
- change:
    - folder_path: folder where the workflow results are saved
    - output_folder: folder to save the results
    - sim: set True for simulated datasets and False for kaggle datasets