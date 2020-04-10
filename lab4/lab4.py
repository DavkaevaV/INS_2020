import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import *

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def build_model(optimizer):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])
    return model
	
def loadImg(path):
    img = Image.open(path)
    img = img.resize((28, 28))
    img = np.dot(np.asarray(img), np.array([1/3, 1/3, 1/3]))
    img /= 255
    img = 1 - img
    return img.reshape((1, 28, 28))
	
def prediction(path, model):
    return np.argmax(model.predict(loadImg(path)))
	
model = build_model('adam')

h = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_split=0.1)
print(h.history.keys())
epochs = range(1, len(h.history['loss']) + 1)
plt.plot(epochs, h.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, h.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, h.history['accuracy'], 'bo', label='Training acc')
plt.plot(epochs, h.history['val_accuracy'], 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

print(prediction("/2.jpg", model))

optimizers = [RMSprop, Adam, Adagrad, Adadelta, SGD]
lrates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
for lrate in lrates:
    acc = []
    for i in range(len(optimizers)):
        model = build_model(optimizers[i](learning_rate=lrate))
        history = model.fit(train_images, train_labels, epochs=5, batch_size=128)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        acc.append(test_acc)
    labeles_x = [o.__name__ for o in optimizers]
    plt.bar(labeles_x, acc, width=1)
    plt.title('learning_rate='+str(lrate))
    plt.ylabel('Accuracy')
    plt.show()
    plt.clf()
