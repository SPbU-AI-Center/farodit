import pandas as pd


def load_csv(date_column, columns):
    file_name = 'data.csv'
    df = pd.read_csv(file_name)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    df = df[columns].dropna()

    return df[columns]
