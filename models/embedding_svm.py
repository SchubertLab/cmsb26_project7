import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.dataloader as dl
import lib.datasplit as ds
import lib.metrics as me
# import encoding.kmer_freq as kf
import encoding.tcr_bert as tcrbert
import pandas as pd
import numpy as np


if not os.path.exists("./data/simulated_2k_balanced_noisy_25_dataset"):
    print("Computing TCR-BERT embeddings...")
    
    df = dl.load_airr_dataset("simulated_2k_balanced_noisy_25_dataset")
    df_merged = dl.merge_dataset(df)

    df_embedding = tcrbert.encode_repertorie(
        df_merged,
        sequence_col="cdr3_aa",
        max_length=64,
        tcr_agg_method="mean",
        repertoire_agg_method="mean"
    )

    print (f"Encoded {len(df_embedding)} samples with TCR-BERT embeddings.")
    print(df_embedding.head())
    
    os.makedirs("./data/simulated_2k_balanced_noisy_25_dataset", exist_ok=True)
    df_embedding.to_pickle("./data/simulated_2k_balanced_noisy_25_dataset/df_embedding.pkl")
else:
    df_embedding = pd.read_pickle("./data/simulated_2k_balanced_noisy_25_dataset/df_embedding.pkl")
    print("Loaded precomputed embeddings.")
    print(df_embedding.head())


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# X = np.vstack(df_embedding["embeddings"].values)

X = np.vstack([
    emb.detach().cpu().numpy()
    for emb in df_embedding["embeddings"].values
])

y = df_embedding["disease"].values

print("Feature matrix shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="linear",
        class_weight="balanced",
        probability=True
    ))
])

pipe_svm.fit(X_train, y_train)

# predict on test set
y_pred = pipe_svm.predict(X_test)

print(y_pred)

metrics = me.calc_metrics(y_test, y_pred)

print(metrics)
