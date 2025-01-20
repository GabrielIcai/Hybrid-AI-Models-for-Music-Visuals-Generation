import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    data=pd.read_csv(data_path,encoding='latin-1')
    return data

def split_dataset(data, test_size=0.2, random_state=42):
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, val_data