import flower.sklearn.utils as utils

if __name__ == "__main__":
    # Load UNSW
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    print(X_train.shape)
    print(y_train.shape)

    (X_train, y_train), (X_test, y_test) = utils.load_unsw()

    print(X_train.shape)
    print(y_train.shape)