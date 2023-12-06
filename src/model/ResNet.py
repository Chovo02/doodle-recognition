from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Reshape

class ResNet:
    def __init__(self) -> None:
        
        X_input = Input((28, 28))
        X = Reshape(target_shape=(28, 28, 1))(X_input)
        X = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)

        X = self.identity_block(X, f=3, filters=[64, 64], stage=2, block='a')
        X = self.identity_block(X, f=3, filters=[64, 64], stage=2, block='b')

        X = Flatten()(X)
        X = Dense(345, activation='softmax', name='fc' + str(354))(X)

        self.model = Model(inputs=X_input, outputs=X, name='ResNet')
        
        self.callback = [
            EarlyStopping(patience=3)
        ]
    
    def build(self):
        self.model.build(input_shape=(28, 28))
        
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
        
    def identity_block(self, X, f, filters, stage, block):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2 = filters

        X_shortcut = X

        X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X
