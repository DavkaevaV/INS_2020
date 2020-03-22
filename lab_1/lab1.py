import random
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
rand = list(range(len(dataset)))
random.seed(999)
random.shuffle(rand)
dataset = dataset[rand]
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y) #выделение классов
encoded_Y = encoder.transform(Y) #трансформация в числовой аналог
dummy_y = to_categorical(encoded_Y) #Преобразует в двоичную матрицу классов


model = Sequential()

model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

history_dict = history.history
#график ошибки
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#график точности
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
