import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import lib.dataloader as dl
import lib.datasplit as ds
import lib.metrics as me
import encoding.kmer_freq as kf
import encoding.tcr_bert as tb

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




datasets_3AA = ['simulated_200_balanced_dataset', 'simulated_200_unbalanced_dataset', 'simulated_500_balanced_dataset', 'simulated_500_unbalanced_dataset', 'simulated_1k_balanced_dataset', 'simulated_1k_unbalanced_dataset', 'simulated_2k_balanced_dataset', 'simulated_2k_unbalanced_dataset', 'simulated_2k_balanced_noisy_05_dataset', 'simulated_2k_balanced_noisy_25_dataset']
# control: HNDYSEIRCVLQN, disease: GPKALMVFWQRST
datasets_13AA = [str.replace(data,'dataset', '13AA_dataset') for data in datasets_3AA]

# kaggle datasets
datasets_kaggle = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5', 'dataset_6', 'dataset_7_1', 'dataset_7_2', 'dataset_8_1', 'dataset_8_2', 'dataset_8_3', 'dataset_7', 'dataset_8']

# new simulated datasets
datasets_test = ['simulated_1000_balanced_test_no_noisy_13_AA', 'simulated_1000_balanced_test_noisy_13_AA']

sample_column='sample'
label_column='disease'
sequence_column = 'cdr3_aa'

for data in datasets_3AA[:1]:
    df = dl.load_airr_dataset(data)

    #df = dl.load_kaggle_dataset(data)

    df = df[[sample_column, label_column, sequence_column]]
    print(df.shape)
    print(df.head())

    df_encode = tb.encode_sequences(df, seq_col=sequence_column, max_length=32, batch_size=1024)

    _, _, embeddings = df_encode

    columns = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df_new = pd.DataFrame(embeddings.detach().cpu().numpy(), columns=columns)

    print(df_new.head())
    print(df_new.shape)

    df_merged = pd.concat([df, df_new], axis=1)

    print(df_merged.head())
    print(df_merged.shape)
