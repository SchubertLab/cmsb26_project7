from collections import defaultdict
import pandas as pd
from collections import Counter

def encode_repertories(df, k=3, sequence_column="cdr3_aa", sample_column="label", label_column="disease"):
    # Step 1: find all unique kmers
    kmer_set = set()
    for seq in df[sequence_column]:
        seq = seq.upper()
        kmer_set.update([seq[i:i+k] for i in range(len(seq) - k + 1)])
    kmers = sorted(kmer_set)
    
    # Step 2: aggregate sequences per sample
    print("Calculating k-mer Frequencies...")
    aggregated_data = []
    for sample, group in df.groupby(sample_column):
        label = group[label_column].iloc[0]  # assume label is same for all sequences of the sample
        kmer_counts = Counter()
        for seq in group[sequence_column]:
            seq = seq.upper()
            kmer_counts.update([seq[i:i+k] for i in range(len(seq) - k + 1)])
        row_data = [sample, label] + [kmer_counts.get(kmer, 0) for kmer in kmers]
        aggregated_data.append(row_data)
    
    # Step 3: create final DataFrame
    print("Creating final DataFrame...")
    encoded_df = pd.DataFrame(aggregated_data, columns=[sample_column, label_column] + kmers)
    
    return encoded_df
