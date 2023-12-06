from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape, Input
import numpy as np

class LeNet5:
    def __init__(self) -> None:
        self.model = Sequential(
            [
                Input(shape=(28, 28)),
                Reshape(target_shape=(28, 28, 1)),
                Conv2D(filters=6, kernel_size=5, padding="same", activation="sigmoid"),
                MaxPooling2D(pool_size=2, strides=2),
                Conv2D(filters=16, kernel_size=5, padding="same", activation="sigmoid"),
                MaxPooling2D(pool_size=2, strides=2),
                Flatten(),
                Dense(units=1200, activation="sigmoid"),
                Dense(units=840, activation="sigmoid"),
                Dense(units=345, activation="softmax"),
            ]
        )
        self.callback = [
            EarlyStopping(patience=3)
        ]
        self.class_array = np.load("data/class.npy")
        
    
    def build(self):
        self.model.build()
        
    def compile(self):
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(
            x=x_train,
            y=y_train,
            epochs=100,
            batch_size=2048,
            validation_data=(x_val, y_val),
            callbacks=self.callback,
            workers=-1
        )
    
    def save_weights(self, path):
        self.model.save_weights(path)
        
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def predict(self, x):
        prediction  = self.model.predict(x, verbose=0)[0]
        top = np.argsort(prediction)[-5:][::-1]
        top_encoded = self.class_array[top]
        return prediction[top], top_encoded