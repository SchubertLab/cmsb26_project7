import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import preprocess_data, metric_heatmap
from models.random_forest import RandForestPredictor

import pickle


if __name__ == '__main__':
    # Load later
    with open('output_stats/test/random_forest_variables.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data.keys())

    for key, value in data.items():
        pass
    

