from collections import defaultdict


def encode_repertories(df, k=5, sequence_column="cdr3_aa", sample_column="label", label_column="disease"):

    dict_samples = {}
    for sample_id, group in df.groupby(sample_column):
        sequences = group[sequence_column].tolist()
        kmer_freq_dict = kmer_frequency(sequences, k)
        dict_samples[sample_id] = kmer_freq_dict
    return dict_samples
    

# take list of sequences and return k-mer frequency dictionary
def kmer_frequency(sequences, k):

    kmer_freq = defaultdict(int)
    total_kmers = 0

    for seq in sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            kmer_freq[kmer] += 1
            total_kmers += 1

    # Convert counts to frequencies
    for kmer in kmer_freq:
        kmer_freq[kmer] /= total_kmers

    return dict(kmer_freq)

if __name__ == "__main__":
    pass