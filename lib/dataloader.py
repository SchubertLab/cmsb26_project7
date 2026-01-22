from pathlib import Path
import pandas as pd 
import yaml 


def load_airr_dataset(dataset_name, save_df=False):

    with open('airr_datasets.yaml', 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.SafeLoader)
    
    if dataset_name not in yaml_file:
        raise ValueError(f"Dataset {dataset_name} not found in airr_datasets.yaml")
    
    dataset_path = Path(yaml_file[dataset_name]['path'])

    metadata = load_metadata(dataset_path)
    df = load_repertoires(dataset_path, metadata)

    if save_df:
        df.to_csv(f"{dataset_name}_airr_dataset.csv", index=False)

    return df


def load_metadata(dataset_path):
    metadata_path = dataset_path / 'simulated_dataset.csv'
    metadata = pd.read_csv(metadata_path)
    return metadata

def load_repertoires(dataset_path, metadata):
    repertoires = []

    for _, row in metadata.iterrows():
        rep_id = row["filename"].replace(".tsv", "")

        tsv = dataset_path / f"{rep_id}.tsv"
        yml = dataset_path / f"{rep_id}.yaml"

        df = pd.read_csv(tsv, sep="\t")
        label = row["identifier"]

        df["label"] = label
        df["repertoire_id"] = rep_id

        repertoires.append(df)

    df = pd.concat(repertoires, ignore_index=True)
    df = df.merge(metadata, left_on='label', right_on='identifier', how='left')
    return df




