from farodit.polynomial_model import PolynomialModel
from airfoil_data_preparation import prepare_airfoil_data

if __name__ == "__main__":
    poly_net = PolynomialModel(order=3, output_dim=1)
    X_train, X_test, Y_train, Y_test = prepare_airfoil_data()
    poly_net.fit(X_train, Y_train, num_epoch=15000, batch_size=500)
    poly_net.validate(X_test, Y_test)