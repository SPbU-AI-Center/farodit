import numpy as np
from filterpy.kalman import KalmanFilter as kf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


class KalmanModel:
    def fit(self, dim_x, dim_z,
            transition_matrix,
            observation_matrix,
            process_covariance,
            observation_covariance,
            initial_state,
            initial_covariance
            ):
        self.model = kf(dim_x, dim_z)

        # F - матрица процесса
        self.model.F = transition_matrix
        # Матрица наблюдения
        self.model.H = observation_matrix
        # Ковариационная матрица ошибки модели
        self.model.Q = process_covariance
        measurementSigma = 0.5
        # Ковариационная матрица ошибки измерения
        self.model.R = observation_covariance
        # Начальное состояние.
        self.model.x = initial_state
        # Ковариационная матрица для начального состояния
        self.model.P = initial_covariance

    def predict(self, X_data):
        filtered_state = np.zeros((len(X_data), self.model.dim_x))
        state_covariance_history = np.zeros((len(X_data), self.model.dim_x, self.model.dim_x))

        for i in range(0, len(X_data)):
            z = np.array([X_data[i]])
            self.model.predict()
            self.model.update(z)

            filtered_state[i] = self.model.x
            state_covariance_history[i] = self.model.P

        return filtered_state, state_covariance_history

    def validate(self, X_data, Y_data):
        Y_model, _ = self.predict(X_data)

        mse = mean_squared_error(Y_data, Y_model[:, 0])
        mae = mean_absolute_error(Y_data, Y_model[:, 0])
        mape = mean_absolute_percentage_error(Y_data, Y_model[:, 0])

        print('MSE  = ', mse)
        print('MAE  = ', mae)
        print('MAPE = ', mape)

        t = range(len(X_data))
        plt.scatter(t, Y_data, label="Истинные значения")
        plt.scatter(t, Y_model[:, 0], label="Прогнозируемые значения")
        plt.legend()
        plt.show()

