from pathlib import Path
import pandas as pd 
import yaml 


def load_airr_dataset(dataset_name):

    with open('/vol/data/immuneML/cmsb26_project7/lib/airr_datasets.yaml', 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.SafeLoader)
    
    if dataset_name not in yaml_file:
        raise ValueError(f"Dataset {dataset_name} not found in airr_datasets.yaml")
    
    if (Path(yaml_file[dataset_name]['path']) / "_dataset.pkl").exists():
        print(f"Loading cached dataset from {yaml_file[dataset_name]['path']}/_dataset.pkl...")
        df = pd.read_pickle(Path(yaml_file[dataset_name]['path']) / "_dataset.pkl")
        return df
    
    dataset_path = Path(yaml_file[dataset_name]['path'])

    metadata = load_metadata(dataset_path)
    df = load_repertoires_airr(dataset_path, metadata)
    
    print(f"Saving merged dataset to {dataset_path}/_dataset.pkl...")
    df.to_pickle(dataset_path / "_dataset.pkl")

    return df

def load_kaggle_dataset(dataset_name):
    
    with open('/vol/data/immuneML/cmsb26_project7/lib/kaggle_datasets.yaml', 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.SafeLoader)
        
    if dataset_name not in yaml_file:
        raise ValueError(f"Dataset {dataset_name} not found in kaggle_datasets.yaml")
    
    train_path = yaml_file[dataset_name]["train_path"]
    # test_path = yaml_file[dataset_name]["test_path"]
    
    print(f"Loading Kaggle dataset: {dataset_name}")
    print(f"Train path: {train_path}")
    # print(f"Test path: {test_path}")
    
    # check if train_path + merged_dataset.pkl exists
    if (Path(train_path) / "_dataset.pkl").exists():
        print(f"Loading cached dataset from {train_path}/_dataset.pkl...")
        df = pd.read_pickle(Path(train_path) / "_dataset.pkl")
        return df
    
    # load train dataset
    print(f"Loading train dataset from {train_path}...")
    metadata_train = load_metadata(train_path, "metadata.csv")
    df_train = load_repertoires_kaggle(train_path, metadata_train)
    
    # load test dataset
    # print(f"Loading test dataset from {test_path}...")
    # metadata_test = load_metadata(test_path, "metadata.csv")
    # df_test = load_repertoires_kaggle(test_path, metadata_test)
    # df_test["set"] = "test"
    
    # concatenate train and test datasets
    # df = pd.concat([df_train, df_test], ignore_index=True)
    
    df = df_train  # only train for now (test does not contain labels)
    
    # remove columns not needed
    cols_to_remove = ['A', 'B', 'C', 
                      'DPA1', 'DPB1', 'DQA1', 'DQB1', 'DRB3', 'DRB4', 'DRB5', 'DRB1', 
                      'repertoire_id_x', 'repertoire_id_y', 'v_call', 'j_call', 'd_call']
    df = df.drop(columns=cols_to_remove, errors='ignore')
    
    # save merged dataset
    print(f"Saving merged dataset to {train_path}/_dataset.pkl...")
    df.to_pickle(Path(train_path) / "_dataset.pkl")
    
    return df
    


def merge_dataset(df, sample_col="sample", sequence_col="cdr3_aa", label_col="disease", save_df=False):
    df_merged = (
        df
        .groupby(sample_col, as_index=False)
        .agg({
            sequence_col: list,
            label_col: "first"   # or any consistent rule
        })
    )
    return df_merged
    


def load_metadata(dataset_path, file_name="simulated_dataset.csv"):
    if type(dataset_path) is str:
        dataset_path = Path(dataset_path)
    
    metadata_path = dataset_path / file_name
    metadata = pd.read_csv(metadata_path)
    return metadata

def load_repertoires_airr(dataset_path, metadata):
    repertoires = []

    for _, row in metadata.iterrows():
        rep_id = row["filename"].replace(".tsv", "")

        tsv = dataset_path / "repertoires" / f"{rep_id}.tsv"
        yml = dataset_path / "repertoires" / f"{rep_id}.yaml"

        df = pd.read_csv(tsv, sep="\t")
        sample = row["identifier"]

        df["sample"] = sample
        df["repertoire_id"] = rep_id

        repertoires.append(df)

    df = pd.concat(repertoires, ignore_index=True)
    df = df.merge(metadata, left_on='sample', right_on='identifier', how='left')
    return df

def load_repertoires_kaggle(dataset_path, metadata):
    repertoires = []

    for _, row in metadata.iterrows():
        rep_id = row["filename"].replace(".tsv", "")

        tsv = Path(dataset_path) / f"{rep_id}.tsv"

        df = pd.read_csv(tsv, sep="\t")
        sample = rep_id

        df["sample"] = sample
        df["repertoire_id"] = rep_id

        repertoires.append(df)

    df = pd.concat(repertoires, ignore_index=True)
    df = df.merge(metadata, left_on='sample', right_on='repertoire_id', how='left')
    return df


# main function for testing
if __name__ == "__main__":
    dataset_name = "simulated_200_balanced_dataset"
    df = load_airr_dataset(dataset_name)
    print(df)


