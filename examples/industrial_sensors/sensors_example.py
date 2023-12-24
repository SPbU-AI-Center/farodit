from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from farodit.predictor import Predictor
from tags import base_tags, sensor_22_predictors_tags
from load import load_csv
from plot import plot

window_size = 12
future_response_size = 48
test_dataset_size_ratio = 0.3
timestamp_column_name = 'DATE'
target_sensor_name = 'sensor_22'
target_sensor = [target_sensor_name]
predictor_columns = sensor_22_predictors_tags
columns_to_load = [timestamp_column_name] + base_tags

# load data
dataset = load_csv(date_column=timestamp_column_name, columns=columns_to_load)

# regressor test
ridge_regression_model = Ridge(alpha=0.01, tol=1e-3, solver='auto')
predictor = Predictor(ridge_regression_model, target_sensor_name, predictor_columns, is_using_previous_y=False)
predictor.set_future_points_count(future_response_size)
predictor.set_sliding_window_size(window_size)
predictor.set_outliers_filter_settings(upper_percentile=98, lower_persentile=2, lower_bound=8)
predictor.set_mean_filter_settings(mean_window_size=5)

training_dataset, testing_dataset = train_test_split(dataset, test_size=test_dataset_size_ratio, shuffle=False)
training_dataset.reset_index(inplace=True, drop=True)
testing_dataset.reset_index(inplace=True, drop=True)
predicted_testing_dataset = testing_dataset
predictor.fit(training_dataset)
predicted_training = predictor.predict(training_dataset)
predicted_dataframe = predictor.predict(predicted_testing_dataset)

print('test set shape', testing_dataset.shape)
print('predicted shape', predicted_dataframe.shape)
plot(training_dataset.loc[window_size - 1:, target_sensor_name].values,
     predicted_training[:, -future_response_size],
     predicted_testing_dataset.loc[window_size - 1:, target_sensor_name].values,
     predicted_dataframe[:, -future_response_size],
     f'windowed ({window_size}) {target_sensor_name} mae %.2f rmse %.2f',
     show_plot=True,
     save_plot=True)
