from collections import defaultdict


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