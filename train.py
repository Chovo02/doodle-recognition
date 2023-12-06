import numpy as np
from src.DataPreparation import Dataset
from src.Logging import Log
from src.model.LeNet5 import LeNet5
from src.model.AlexNet import AlexNet
from src.model.VGGNet import VGGNet
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import os

log = Log()
dataset = Dataset(log)

x, y = dataset.get_dataset(20_000)

x_train, x_val, y_train, y_val = dataset.get_train_test_split(x, y)
encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)
if not os.path.exists("data/class.npy"):
    np.save("data/class.npy", encoder.classes_)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

for model in [LeNet5, AlexNet, VGGNet]:
    log.write_log(f"Training {model.__name__}")
    model = model()
    model.build()
    model.compile()
    model.fit(x_train, y_train, x_val, y_val)
    model.save_weights(f"weights/{model.__name__}.h5")
    log.write_log(f"Finished training {model.__name__}")