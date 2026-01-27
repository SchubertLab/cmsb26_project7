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

np.set_printoptions(threshold=10000)


if not os.path.exists("./data/dataset_7"):
    print("Computing TCR-BERT embeddings...")
    
    df = dl.load_kaggle_dataset("dataset_7")
    
    
    # for testing only use 1000th of the data
    df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    
    # save testing dataframe
    df.to_csv("./df_sampled.csv", index=False)
    
    # load data
    df = pd.read_csv("./df_sampled.csv")
    
    input_ids, attention_masks, embeddings = tcrbert.encode_sequences(df, seq_col="junction_aa", max_length=32, batch_size=int(8192/2))
    
    # save as npz file
    np.savez_compressed(
        "./embeddings_tcrbert.npz",
        input_ids=input_ids,
        attention_masks=attention_masks,
        embeddings=embeddings
    )
 
    
else:
    pass


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from scipy.spatial.distance import pdist, squareform


print(input_ids.shape)
print(input_ids)

print(input_ids[0])
print(input_ids[-1])

print(attention_masks.shape)
print(attention_masks)

print(embeddings.shape)
print(embeddings)

dist_matrix = squareform(pdist(embeddings*10000, metric="cosine"))

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.imshow(dist_matrix)
plt.colorbar(label="Distance")
plt.title("Pairwise Embedding Distances")
plt.xlabel("Sample index")
plt.ylabel("Sample index")
plt.tight_layout()
# save figure
plt.savefig("pairwise_embedding_distances.png", dpi=300)
# plt.show()

# plot histogram of distances
# log scale y axis
plt.figure(figsize=(6, 4))
plt.hist(dist_matrix.flatten(), bins=80, color='blue', alpha=0.7, edgecolor='black')
plt.title("Histogram of Pairwise Embedding Distances")
plt.xlabel("Distance")
# plt.xscale("log")
plt.ylabel("Frequency")
plt.yscale("log")
plt.tight_layout()
plt.savefig("histogram_embedding_distances.png", dpi=300)
# plt.show()




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
