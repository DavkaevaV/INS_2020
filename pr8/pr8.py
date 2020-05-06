import numpy as np
from datetime import datetime

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from var4 import gen_data
import random


class call_back(Callback):
  def __init__(self, epochs, prefix='model', date=datetime.now()):
    self.prefix = '{}_{}_{}_'.format(date.day, date.month,date.year)+prefix+'_'
    self.epochs = epochs

  def on_epoch_end(self, epoch, logs=None):
    if epoch in self.epochs:
      self.model.save(self.prefix + str(epoch))

def draw_graphics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'g')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()

def getData():
    data, labels = gen_data(size=1000)
    data, labels = shuffle(data, labels)
    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, labels, test_size=0.2, random_state=11)
    dataTrain = dataTrain.reshape(dataTrain.shape[0], 50, 50, 1)
    dataTest = dataTest.reshape(dataTest.shape[0], 50, 50, 1)

    encoder = LabelEncoder()
    encoder.fit(labelTrain)
    labelTrain = encoder.transform(labelTrain)
    labelTrain = to_categorical(labelTrain, 2)

    encoder.fit(labelTest)
    labelTest = encoder.transform(labelTest)
    labelTest = to_categorical(labelTest, 2)
    return dataTrain, labelTrain, dataTest, labelTest

def createModel():
    epochs = [1, 3, 7, 8, 9]
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adagrad',
                  metrics=['accuracy'])
    return model

dataTrain, labelTrain, dataTest, labelTest = getData()
model = createModel()
history = model.fit(dataTrain, labelTrain, batch_size=20, epochs=9, verbose=1, validation_data=(dataTest, labelTest), callbacks=[call_back(epochs, 'prefix')])
score = model.evaluate(dataTest, labelTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

draw_graphics(history)