

## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
# check LSTM from scratch.py for the LSTM model coded from scratch!!!
# check LSTM from scratch.py for the LSTM model coded from scratch!!!
# check LSTM from scratch.py for the LSTM model coded from scratch!!!

# for custom loss function, keras library was used as it is difficult 
# to build custom loss functions and test them without automatic differentiation

import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from keras import backend as K


def customLoss(y_pred, y_true):
    # y_pred should be a list of numbers. Each representing how many shares to buy/sell each day
    # negative values for selling, positive for buying.
    capital = 100-K.sum(y_true * y_pred)
    # sell all residual shares to get capital on last day
    capital += (K.sum(y_pred) * y_true[..., -1])
    ROI = ((capital - 100) / 100)
    return ROI * -1


def scaleMat(X):
    numer = (X-X.min(axis=0))
    denom = X.max(axis=0) - X.min(axis=0)
    return numer/denom


df = pd.read_csv('L:\\stonks\\ADANIGREEN.NS-indicators.csv')
df.drop('Date', inplace=True, axis=1)
df = df.apply(pd.to_numeric)
print(df)

X = df.to_numpy()
Y = df[["Close"]].to_numpy()
#Y = df[["close"]].to_numpy()
X = scaleMat(X)
Y = (Y-np.amin(Y)) / \
    (np.amax(Y)-np.amin(Y))
split = int(X.shape[0]/2)
x_train = X[:split]
x_train = np.expand_dims(x_train, axis=(0))
x_test = X[split: -1]
x_test = np.expand_dims(x_test, axis=(0))
Y_train = Y[1: split + 1]
Y_test = Y[split + 1:]

Y_train = Y_train.T
Y_test = Y_test.T
print(Y_train[0, -1])


model = Sequential()

model.add(LSTM(units=50, return_sequences=True, batch_input_shape=(x_train.shape[0],
                                                                   x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss=customLoss)

K.set_value(model.optimizer.learning_rate, 0.0001)

model.fit(x_train, Y_train, epochs=1000)


# model.save('L:\\stonks\\lstm_custom_loss')

predictions = model.predict(x_test).squeeze()
print(predictions)
predictions = (predictions-np.amin(predictions)) / \
    (np.amax(predictions)-np.amin(predictions))

plt.plot(range(1, split), Y_test.squeeze(),
         label="stock closing value (minmax scaled)")
plt.plot(range(1, split), predictions,
         label="amount of shares to buy/sell")
plt.show()
