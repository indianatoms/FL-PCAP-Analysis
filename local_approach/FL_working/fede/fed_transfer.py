class Fed_Avg_Client:
    def __init__(self, name, X_train, y_train, dataset_size, sample_weights):
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.dataset_size = dataset_size
        self.sample_weights = sample_weights