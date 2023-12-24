import numpy as np
from pandas import DataFrame
from typing import List
from farodit.window_slider import WindowSlider

delta_t_name = '∆t'


class Predictor:
    _date_column = 'DATE'

    def __init__(self, model, target: str, columns: List[str], is_using_previous_y: bool):
        self._upper_percentile = 100
        self._lower_persentile = 0
        self._lower_bound = -1000000
        self._mean_window_size = 5
        self._sliding_window_size = 2
        self._future_points_count = 48
        self._max_train_points_count = 3000
        self._is_using_previous_y = is_using_previous_y

        self._target = target
        self._columns = columns
        self._model = model

    def set_using_previous_y(self, is_using_previous_y: bool):
        self._is_using_previous_y = is_using_previous_y

    def set_future_points_count(self, count: int):
        self._future_points_count = count

    def set_max_train_points_count(self, maxcount: int):
        self._max_train_points_count = maxcount

    def set_sliding_window_size(self, size: int):
        self._sliding_window_size = size

    def set_outliers_filter_settings(self, upper_percentile: float, lower_persentile: float, lower_bound: float):
        self._upper_percentile = upper_percentile
        self._lower_persentile = lower_persentile
        self._lower_bound = lower_bound

    def set_mean_filter_settings(self, mean_window_size: int):
        self._mean_window_size = mean_window_size

    def fit(self, df: DataFrame):
        # make copy
        df = df.copy()

        # filter by columns and move target to the end
        df = df[[self._date_column] + self._columns + [self._target]]

        # remove na values
        df.dropna(inplace=True)

        # filter outliers
        df = self._filter_outliers(df)

        # construct deltaT
        df = self._add_delta_t(df)

        # filter rollilng mean
        self._filter_rolling_mean(df)

        # construct windows
        window_constructor = WindowSlider(response_size=self._future_points_count)
        df = window_constructor.collect_windows(df,
                                                window_size=self._sliding_window_size,
                                                previous_y=self._is_using_previous_y)

        # filter windows
        self._filter_windows(df)
        df = self._remove_delta_t_columns(df)

        df = df.tail(self._max_train_points_count)

        # fit
        train_x = df.iloc[:, :-self._future_points_count]
        train_y = df.iloc[:, -self._future_points_count:]

        self._model.fit(train_x, train_y)

    def predict(self, df: DataFrame):
        # make copy
        df = df.copy()

        # filter by columns and move target to the end
        df = df[[self._date_column] + self._columns + [self._target]]

        # construct deltaT
        df = self._add_delta_t(df)

        # construct windows
        window_constructor = WindowSlider(response_size=0)
        df = window_constructor.collect_windows_for_prediction(df,
                                                               window_size=self._sliding_window_size,
                                                               previous_y=self._is_using_previous_y)

        df = self._remove_delta_t_columns(df)
        prediction = self._model.predict(df)

        return prediction

    def _filter_outliers(self, df: DataFrame):
        upper_limit = np.percentile(df[self._target].values, self._upper_percentile)
        lower_limit = max(np.percentile(df[self._target].values, self._lower_persentile), self._lower_bound)
        return df[(df[self._target] < upper_limit) & (df[self._target] > lower_limit)]

    def _add_delta_t(self, df: DataFrame):
        if len(df.index) > 1:
            dates = df[self._date_column]
            deltaT = np.array(
                [(dates.values[i + 1] - dates.values[i]) / np.timedelta64(1, 's') for i in range(len(df) - 1)])
            deltaT[0] = 300
            deltaT = np.concatenate((np.array([300]), deltaT))
        else:
            deltaT = np.array([300])

        df.insert(1, delta_t_name, deltaT)
        needed_columns = [name for name in df.columns if name != self._date_column]
        return df[needed_columns]

    def _filter_rolling_mean(self, df: DataFrame):
        header = df.columns
        df[header[1:]] = df[header[1:]].rolling(self._mean_window_size).mean()
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)

    def _filter_windows(self, windowed_dataframe: DataFrame):
        for name in windowed_dataframe.columns:
            if delta_t_name in name:
                windowed_dataframe.loc[np.abs(windowed_dataframe[name] - 300) > 20, name] = np.nan

        windowed_dataframe.dropna(inplace=True)
        windowed_dataframe.reset_index(inplace=True, drop=True)

    def _remove_delta_t_columns(self, windowed_dataframe: DataFrame):
        columns_without_delta_t = []
        for name in windowed_dataframe.columns:
            if delta_t_name not in name:
                columns_without_delta_t.append(name)

        return windowed_dataframe[columns_without_delta_t]
