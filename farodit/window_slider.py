import numpy as np
import pandas as pd
from pandas import DataFrame


class WindowSlider(object):

    def __init__(self, window_size: int = 5, response_size: int = 1):
        '''
        window_size - number of time steps to look back
        response_size - number of time steps to predict
        slide_length - maximum length to slide - (#observation - w)
        p: final predictors - (#predictors * w)
        '''
        self.window_size = window_size
        self.response_size = response_size
        self.slide_length = 0
        self.predictor_count = 0
        self.names = []

    def collect_windows(self, df: DataFrame, window_size: int = 5, previous_y: bool = False):
        '''
        Input: df датафрейм, первая колонка -- дельта по времени, затем предикторы и цель в последней колонке
        '''
        self.window_size = window_size

        column_count = len(df.columns)
        row_count = len(df.index)

        self.slide_length = row_count - (self.window_size + self.response_size) + 1

        if previous_y:
            self.predictor_count = column_count * (self.window_size)
            X = df
        else:
            self.predictor_count = (column_count - 1) * (self.window_size)
            X = df.iloc[:, :-1]
        # Составляем колонки для предикторов
        columns = X.columns.values
        for i in range(self.window_size):
            for column in columns:
                name = f'{column}({i + 1})'
                self.names.append(name)

        # Составляем колонки для целей
        predictor_name = df.columns[-1]
        for i in range(self.response_size):
            name = f'{predictor_name}_resp({i + 1})'
            self.names.append(name)

        # Инициализация фрейма с окнами
        new_df = pd.DataFrame(np.zeros(shape=(self.slide_length, (self.predictor_count + self.response_size))),
                              columns=self.names)

        # Заполнение фрейма с окнами
        original_frame_predictor_count = len(X.columns)
        if self.response_size > 0:
            for i in range(self.slide_length):
                # Значения предикторов (окно назад)
                predictor_values = X.values[i:i + self.window_size, :original_frame_predictor_count]

                # Целевые значения (окно вперёд)
                y_row = self.window_size + i
                target_values = df.values[y_row:y_row + self.response_size, -1]

                # Построчное заполнение фрейма
                new_df.iloc[i, :] = np.concatenate([predictor_values.flatten(), target_values.flatten()])
        else:
            for i in range(self.slide_length):
                # Значения предикторов (окно назад)
                predictor_values = X.values[i:i + self.window_size, :original_frame_predictor_count]

                new_df.iloc[i, :] = predictor_values.flatten()

        return new_df

    def collect_windows_for_prediction(self, df: DataFrame, window_size: int = 5, previous_y: bool = False):
        old_size = self.response_size
        self.response_size = 0

        new_df = self.collect_windows(df, window_size=window_size, previous_y=previous_y)

        self.response_size = old_size

        return new_df
