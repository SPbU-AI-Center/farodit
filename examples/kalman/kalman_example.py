import numpy as np
from filterpy import common
from matplotlib import pyplot as plt

from farodit.kalman_model import KalmanModel

fig, axs = plt.subplots(1, 1, figsize=(12, 10))

transition_matrix = np.array([[1, 1], [0, 1]])
observation_matrix = np.array([[1, 0]])
process_covariance = np.array([[1e-5, 0], [0, 1e-5]])
observation_covariance = np.array([[1e-3]])
initial_state_mean = np.array([0, 0])
initial_state_covariance = np.array([[1, 0], [0, 1]])

dt = 0.01
noiseSigma = 0.5
samplesCount = 1000
noise = np.random.normal(loc=0.0, scale=noiseSigma, size=samplesCount)

trajectory = np.zeros((3, samplesCount))

position = 0
velocity = 1.0
acceleration = 0.0

for i in range(1, samplesCount):
    position = position + velocity * dt + (acceleration * dt ** 2) / 2.0 + 0.1 * np.sin(
        2 * np.pi * i * dt)  # Добавлено синусоидальное колебание
    velocity = velocity + acceleration * dt + 0.2 * np.cos(
        2 * np.pi * i * dt)  # Добавлено косинусоидальное колебание к скорости
    acceleration = acceleration + 0.1 * np.sin(4 * np.pi * i * dt)  # Добавлено синусоидальное колебание к ускорению

    trajectory[0][i] = position
    trajectory[1][i] = velocity
    trajectory[2][i] = acceleration

measurement = trajectory[0] + noise
processNoise = 1e-4

# F - матрица процесса
F = np.array([[1, dt, (dt**2) / 2],
              [0, 1.0, dt],
              [0, 0, 1.0]])

# Матрица наблюдения
H = np.array([[1.0, 0.0, 0.0]])

# Ковариационная матрица ошибки модели
Q = common.Q_discrete_white_noise(dim=3, dt=dt, var=processNoise)

measurementSigma = 0.5
# Ковариационная матрица ошибки измерения
R = np.array([[measurementSigma * measurementSigma]])

# Начальное состояние.
x = np.array([0.0, 0.0, 0.0])

# Ковариационная матрица для начального состояния
P = np.array([[10.0, 0.0, 0.0],
              [0.0, 10.0, 0.0],
              [0.0, 0.0, 10.0]])

k = KalmanModel()
k.fit(dim_x=3,
      dim_z=1,
      transition_matrix=F,
      observation_matrix=H,
      initial_state=x,
      initial_covariance=P,
      observation_covariance=R,
      process_covariance=Q
      )

filteredState, stateCovarianceHistory = k.predict(measurement)

axs.set_title("Kalman filter (3rd order)")
axs.plot(measurement, label="Измерение")
axs.plot(trajectory[0], label="Истинное значение")
axs.plot(filteredState[:, 0], label="Оценка фильтра")
axs.legend()

plt.tight_layout()
plt.show()
