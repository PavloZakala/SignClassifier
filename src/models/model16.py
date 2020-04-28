from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

def create_model_16():

    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(CLASS_NUMBER, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()