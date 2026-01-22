import sys
import os

# add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.dataloader as dl
import lib.datasplit as ds
import encoding.kmer_freq as kf
import pandas as pd


df = dl.load_airr_dataset("simulated_2k_balanced_dataset")
kmer_freq = kf.encode_repertorie_normalized(df, k=4, sequence_column="cdr3_aa", sample_column="label", label_column="disease")

print (f"Encoded {len(kmer_freq)} samples with k-mer frequencies.")

print(kmer_freq)

train, test = ds.split_data(kmer_freq)

print(train)
print(test)