import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from transformers.file_utils import TRANSFORMERS_CACHE

# print("Hugging Face cache directory:", TRANSFORMERS_CACHE)


model = BertModel.from_pretrained("wukevin/tcr-bert").to(device)
tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert")

model.eval()

print(f"TCR-BERT loaded on {device}")

# Take dataframe as input and add one hot and mask columns
def tokenize(df, seq_col="junction_aa", max_length=32):
    input_ids = []
    attention_masks = []

    for seq in tqdm(df[seq_col], desc="Tokenizing sequences"):
        encoded_dict = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    df["input_ids"] = input_ids
    df["attention_mask"] = attention_masks

    return df

from tqdm import tqdm
import torch

def tokenize_batchwise(df, seq_col="junction_aa", max_length=32, batch_size=1024):
    input_ids = []
    attention_masks = []

    sequences = df[seq_col].tolist()
    
    print(sequences[:20])
    
    # amino acids must be space delimited for TCR-BERT tokenizer
    sequences = [" ".join(list(seq)) for seq in sequences]

    print(sequences[:20])
    
    print(sequences[-20:])

    for i in tqdm(range(0, len(sequences), batch_size), desc="Tokenizing sequences"):
        batch_seqs = sequences[i:i + batch_size]

        encoded = tokenizer(
            batch_seqs,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    # concatenate batches
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return (input_ids, attention_masks)


def encode_sequences(df, seq_col="junction_aa", max_length=32, batch_size=1024):
    input_ids, attention_masks = tokenize_batchwise(df, seq_col=seq_col, max_length=max_length, batch_size=batch_size)
    embeddings = []
    
    
    for i in tqdm(range(0, input_ids.size(0), batch_size), desc="Encoding sequences"):
        batch_input_ids = input_ids[i:i + batch_size].to(device)
        batch_attention_masks = attention_masks[i:i + batch_size].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
            )
            # CLS token
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings.append(batch_embeddings.cpu())

    # shape: (N, hidden_dim)
    embeddings = torch.cat(embeddings, dim=0)

    # df["embeddings"] = list(embeddings)  # one tensor per row
    return (input_ids, attention_masks, embeddings)
    
    