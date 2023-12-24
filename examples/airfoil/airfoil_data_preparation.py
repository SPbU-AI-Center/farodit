import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def load_data():
    filepath = "AirfoilSelfNoise.csv"
    return pd.read_csv(filepath)


def split_data(data_frame):
    X = data_frame[["f", "alpha", "c", "U_infinity", "delta"]]
    Y = data_frame[["SSPL"]]
    return train_test_split(X, Y, test_size=0.33)


def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def prepare_airfoil_data():
    data = load_data()
    X_train, X_test, Y_train, Y_test = split_data(data)
    X_train, X_test = scale_data(X_train, X_test)
    return X_train, X_test, Y_train.values.ravel(), Y_test.values.ravel()
