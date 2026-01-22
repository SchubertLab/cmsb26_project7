from collections import defaultdict
import pandas as pd
from collections import Counter
from tqdm import tqdm

def encode_repertorie_raw(
        df, 
        k=3, 
        sequence_column="cdr3_aa", 
        sample_column="sample", 
        label_column="disease"
    ):  
    
    # Step 1: find all unique kmers
    kmer_set = set()
    for seq in df[sequence_column]:
        seq = seq.upper()
        kmer_set.update([seq[i:i+k] for i in range(len(seq) - k + 1)])
    kmers = sorted(kmer_set)
    
    # Step 2: aggregate sequences per sample
    aggregated_data = []
    
    grouped = df.groupby(sample_column)
    for sample, group in tqdm(grouped, desc="Calculating k-mer Frequencies", total=grouped.ngroups):
        label = group[label_column].iloc[0]  # assume same label per sample
        kmer_counts = Counter()

        for seq in group[sequence_column]:
            seq = seq.upper()
            kmer_counts.update(seq[i:i+k] for i in range(len(seq) - k + 1))

        row_data = [sample, label] + [kmer_counts.get(kmer, 0) for kmer in kmers]
        aggregated_data.append(row_data)
    
    # Step 3: create final DataFrame
    print("Creating final DataFrame...")
    encoded_df = pd.DataFrame(aggregated_data, columns=[sample_column, label_column] + kmers)
    
    return encoded_df

def encode_repertorie_normalized(
        df, 
        k=3, 
        sequence_column="cdr3_aa",
        sample_column="sample", 
        label_column="disease"
    ):
    # Step 1: find all unique kmers
    kmer_set = set()
    for seq in tqdm(df[sequence_column], desc="Finding unique k-mers"):
        seq = seq.upper()
        kmer_set.update(seq[i:i+k] for i in range(len(seq) - k + 1))
    kmers = sorted(kmer_set)

    # Step 2: aggregate + normalize per sample
    aggregated_data = []

    grouped = df.groupby(sample_column)
    for sample, group in tqdm(grouped, desc="Calculating k-mer Frequencies", total=grouped.ngroups):
        label = group[label_column].iloc[0]
        kmer_counts = Counter()
        total_kmers = 0

        for seq in group[sequence_column]:
            seq = seq.upper()
            n = len(seq) - k + 1
            if n > 0:
                total_kmers += n
                kmer_counts.update(seq[i:i+k] for i in range(n))

        # normalize here
        row_data = (
            [sample, label] +
            [(kmer_counts[kmer] / total_kmers) if total_kmers > 0 else 0.0
             for kmer in kmers]
        )
        aggregated_data.append(row_data)

    # Step 3: DataFrame
    encoded_df = pd.DataFrame(
        aggregated_data,
        columns=[sample_column, label_column] + kmers
    )

    return encoded_df