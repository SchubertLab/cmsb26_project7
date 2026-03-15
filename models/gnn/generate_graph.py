import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def top_percent_cutoff(df, p=0.01):
    freq = (
        df.groupby(["sample", "templates"])
        .size()
        .reset_index(name="count")
        .sort_values(by=["sample", "templates"], ascending=True)
    )
    
    freq["cummulative"] = freq.groupby("sample")["count"].cumsum()
    freq["total"] = freq.groupby("sample")["count"].transform("sum")

    freq["cummulative_perc"] = freq["cummulative"] / freq["total"]
    freq.reset_index(drop=True, inplace=True)
    
    df = df.merge(freq[["sample", "templates", "cummulative_perc"]], on=["sample", "templates"], how='left')
    df_filtered = df[df["cummulative_perc"] >= 1-p]
    return df_filtered


def frequency_cutoff(df, min_freq=0.00001):
    df["total_templates"] = df.groupby("sample")["templates"].transform("sum")
    df["template_perc"] = df["templates"] / df["total_templates"]
    df_filtered = df[df["template_perc"] >= min_freq]
    return df_filtered

def naive_sample_node_merge(df_sample, embeddings_sample, k_nodes=500):

    kmeans = KMeans(n_clusters=k_nodes, random_state=42)
    labels = kmeans.fit_predict(embeddings_sample)

    cluster_embeddings = kmeans.cluster_centers_

    # kmeans = MiniBatchKMeans(n_clusters=k_nodes, random_state=42, batch_size=1000)
    # labels = kmeans.fit_predict(embeddings_sample)

    # cluster_embeddings = kmeans.cluster_centers_

    df_sample["cluster"] = labels
    df_sample["cluster_embedding"] = df_sample["cluster"].apply(lambda x: cluster_embeddings[x])

    df_grouped = df_sample.groupby("cluster").agg({
        "templates": "sum",
        "cluster_embedding": "first",  # all rows in a cluster share the same centroid
        "sample": "first",  # optional, pick representative
        "filename": "first",
        "label_positive": "first",
        # "study_group_description": "first",
        "sex": "first",
        "age": "first"
    }).reset_index()

    return df_grouped

def native_node_merge(df, embeddings, k_nodes=500):
    all_grouped = []
    i = 0
    for sample_id in df["sample"].unique():
        # Get indices of rows belonging to this sample
        idx = df["sample"] == sample_id
        df_sample = df[idx].copy()
        embeddings_sample = embeddings[idx]

        df_grouped = naive_sample_node_merge(df_sample, embeddings_sample, k_nodes=k_nodes)
        all_grouped.append(df_grouped)
        i += 1
        if i % 1 == 0:
            print("Samples processed: " + str(i))
    
    df_merged = pd.concat(all_grouped, ignore_index=True)

    return df_merged
    

