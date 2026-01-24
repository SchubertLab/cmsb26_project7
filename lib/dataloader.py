from pathlib import Path
import pandas as pd 
import yaml 


def load_airr_dataset(dataset_name, save_df=False):

    with open('/vol/data/immuneML/cmsb26_project7/lib/airr_datasets.yaml', 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.SafeLoader)
    
    if dataset_name not in yaml_file:
        raise ValueError(f"Dataset {dataset_name} not found in airr_datasets.yaml")
    
    dataset_path = Path(yaml_file[dataset_name]['path'])

    metadata = load_metadata(dataset_path)
    df = load_repertoires(dataset_path, metadata)

    # Todo: save dataframe in path if needed
    # if save_df:
    #     df.to_csv(f"{dataset_name}_airr_dataset.csv", index=False)

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
    


def load_metadata(dataset_path):
    metadata_path = dataset_path / 'simulated_dataset.csv'
    metadata = pd.read_csv(metadata_path)
    return metadata

def load_repertoires(dataset_path, metadata):
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


# main function for testing
if __name__ == "__main__":
    dataset_name = "simulated_200_balanced_dataset"
    df = load_airr_dataset(dataset_name)
    print(df)


