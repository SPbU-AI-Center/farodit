base_tags = sensors = [f'sensor_{i + 1}' for i in range(31)]

future_prediction_tags = [
    'sensor_22',
    'sensor_23'
]

sensor_22_predictors_tags = [tag for tag in base_tags if
                             tag not in ['sensor_22', 'sensor_23', 'sensor_28', 'sensor_29', 'sensor_30', 'sensor_31']]
sensor_23_predictors_tags = [tag for tag in base_tags if tag not in ['sensor_22', 'sensor_23', 'sensor_25']]
