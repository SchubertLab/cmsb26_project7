import lib.dataloader as dl
import lib.datasplit as ds
import encoding.kmer_freq as kf

def prep_data(dataset, k=3, sequence_column="cdr3_aa", sample_column="sample", label_column="disease"):
    df = dl.load_airr_dataset(dataset)
    train, test = ds.split_data(df)
    
    train = kf.encode_repertorie_normalized(train, k=k, sequence_column=sequence_column, sample_column=sample_column, label_column=label_column)
    test = kf.encode_repertorie_normalized(test, k=k, sequence_column=sequence_column, sample_column=sample_column, label_column=label_column)

    # keep only columns present in train
    test = test.reindex(columns=train.columns, fill_value=0)
    
    return train, test