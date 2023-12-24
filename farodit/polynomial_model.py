import time

from keras import Sequential, optimizers
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from farodit import PolynomialNeuralLayer


class PolynomialModel:
    def __init__(self, order, output_dim):
        self.model = Sequential()
        self.model.add(PolynomialNeuralLayer(output_dimension=output_dim, polynomial_order=order))
        optimizer = optimizers.Adam(learning_rate=0.05)
        self.model.compile(loss='mean_squared_logarithmic_error', optimizer=optimizer, metrics=['mae', 'mape'])

    def fit(self, X_train, Y_train, num_epoch=10000, batch_size=500):
        start = time.time()
        self.model.fit(X_train, Y_train, epochs=num_epoch, batch_size=batch_size, shuffle=False, verbose=1)
        print(self.model.get_weights())
        print(f'PNN is built in {time.time() - start} seconds')

    def predict(self, X_data):
        return self.model.predict(X_data)

    def validate(self, X_data, Y_data):
        Y_model = self.predict(X_data)
        print('MSE  = ', mean_squared_error(Y_data, Y_model))
        print('MAE  = ', mean_absolute_error(Y_data, Y_model))
        print('MAPE = ', mean_absolute_percentage_error(Y_data, Y_model))

        t = range(len(X_data))
        plt.scatter(t, Y_data.reshape(1, -1))
        plt.scatter(t, Y_model.reshape(1, -1))
        plt.show()
