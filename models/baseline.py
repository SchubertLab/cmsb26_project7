import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.dataloader as dl
import lib.datasplit as ds
import lib.metrics as me
import encoding.kmer_freq as kf
import pandas as pd
import numpy as np


df = dl.load_airr_dataset("simulated_2k_unbalanced_dataset")
df = kf.encode_repertorie_normalized(df, k=1, sequence_column="cdr3_aa", sample_column="sample", label_column="disease")

print (f"Encoded {len(df)} samples with k-mer frequencies.")

train, test = ds.split_data(df)

print(train)
print(test)

# set random seed for reproducibility
np.random.seed(1)


def class_weighted_baseline(labels, n_samples):
    
    p = np.mean(labels)

    is_positive = np.random.rand(n_samples) < p
    print("Percent positive: " + str(p))

    probs = np.empty(n_samples)
    probs[is_positive] = np.random.uniform(0.5, 1.0, size=is_positive.sum())
    probs[~is_positive] = np.random.uniform(0, 0.5, size=(~is_positive).sum())

    return probs


for i in range(20):
    np.random.seed(i)

    probs = class_weighted_baseline(test["disease"], len(test["disease"]))
    metrics = me.calc_metrics(test["disease"], probs)
    print(metrics)