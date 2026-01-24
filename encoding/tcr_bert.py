from transformers import BertModel
from transformers import BertTokenizer

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertModel.from_pretrained("wukevin/tcr-bert")
tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert")

model.to(device)

print(f"TCR-BERT model loaded on device: {device}")

def tokeninze_tcrs(sequences, max_length=64):
    # space separate the sequences
    spaced_sequences = [" ".join(list(seq)) for seq in sequences]
    
    # add placeholder sequence with same length as max_length to enforce padding/truncation
    placeholder = "A " * (max_length - 1) + "A"
    spaced_sequences.append(placeholder.strip())
    
    inputs = tokenizer(
        spaced_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    # remove the placeholder from inputs
    for key in inputs:
        inputs[key] = inputs[key][:-1]
        
    return inputs

def generate_tcr_embeddings(sequences, max_length=64, device=device):
    inputs = tokeninze_tcrs(sequences, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # get the last hidden states
    embeddings = outputs.hidden_states[-1]
    return embeddings

def aggregate_tcr_embeddings(embedding, tcr_agg_method="mean", repertoire_agg_method="mean"):

    # embedding shape: (batch_size, seq_length, embedding_dim)
    if tcr_agg_method == "mean":
        tcr_embeddings = embedding.mean(dim=1)  # mean over seq_length
    elif tcr_agg_method == "max":
        tcr_embeddings, _ = embedding.max(dim=1)  # max over seq_length
    else:
        raise ValueError(f"Unknown tcr_agg_method: {tcr_agg_method}")

    # Now aggregate over the repertoire (batch dimension)
    if repertoire_agg_method == "mean":
        repertoire_embedding = tcr_embeddings.mean(dim=0)  # mean over batch_size
    elif repertoire_agg_method == "max":
        repertoire_embedding, _ = tcr_embeddings.max(dim=0)  # max over batch_size
    else:
        raise ValueError(f"Unknown repertoire_agg_method: {repertoire_agg_method}")

    return repertoire_embedding

def encode_repertorie(
    df,
    sequence_col="cdr3_aa",
    max_length=64,
    tcr_agg_method="mean",
    repertoire_agg_method="mean",
    device=device,
):
    df["embeddings"] = df.apply(
            lambda row: aggregate_tcr_embeddings(
                generate_tcr_embeddings(row[sequence_col], max_length=max_length, device=device),
                tcr_agg_method=tcr_agg_method,
                repertoire_agg_method=repertoire_agg_method,
        ),
        axis=1
    )
    return df