import sys
import os

# add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.dataloader as dl
import encoding.kmer_freq as kf
import pandas as pd


df = dl.load_airr_dataset("simulated_200_balanced_dataset")