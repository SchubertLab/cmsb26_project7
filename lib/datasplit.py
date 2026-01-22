from sklearn.model_selection import train_test_split 

def split_data(df, test_size=0.2, seed=33):
    return train_test_split(df, test_size=test_size, random_state=seed)

