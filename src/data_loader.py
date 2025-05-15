import pandas as pd

def load_nsl_kdd(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    return df