import numpy as np                                            
import matplotlib.pyplot as plt                       
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from var4 import gen_sequence  

def gen_data_from_sequence(seq_len=1000, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return (past, future)


data, res = gen_data_from_sequence(seq_len=1000, lookback=10)

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]

model = Sequential()
model.add(layers.GRU(32, recurrent_activation='sigmoid', input_shape=(None, 1), return_sequences=True))
model.add(layers.LSTM(32, activation='relu', input_shape=(None, 1), return_sequences=True, dropout=0.2))
model.add(layers.GRU(32, input_shape=(None, 1), recurrent_dropout=0.2))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')


history = model.fit(train_data, train_res, epochs=20, validation_data=(val_data, val_res))

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss)
plt.plot(range(len(val_loss)), val_loss)
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length, predicted_res)
plt.plot(pred_length, test_res)
plt.title('Sequence')
plt.xlabel('x')
plt.ylabel('Sequence')
plt.legend(['Predict', 'Test'], loc='upper right')
plt.show()