import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class RNN:
    def __init__(self, inputSize, outputSize, hiddenSize=75,
                 sequenceLength=500):
        self.inputSize = inputSize
        self.n_h = max(hiddenSize, inputSize)
        self.sequenceLength = sequenceLength
        self.outputSize = outputSize

        # weights
        self.W = np.random.randn(self.inputSize, self.n_h) / 1000
        self.U = np.identity(self.n_h) / 1000
        self.V = np.random.randn(self.n_h, self.outputSize) / 1000

        # biases
        self.bh = np.zeros((1, self.n_h))
        self.by = np.zeros((1, self.outputSize))

        # derivatives
        # weights
        self.dW = np.zeros((self.inputSize, self.n_h))
        self.dV = np.zeros((self.n_h, self.outputSize))
        self.dU = np.zeros((self.n_h, self.n_h))

        # biases
        self.dbh = np.zeros((1, self.n_h))
        self.dby = np.zeros((1, self.outputSize))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def forwardPass(self, X):
        h = []  # stores hidden states
        h.append(np.zeros((1, self.n_h)))  # set initial hidden state to all 0s
        y_pred = []  # stores predictions

        # produce a prediction for each timestep
        for t in range(self.sequenceLength):
            h.append(np.tanh(
                np.dot(X[t], self.W) + np.dot(h[t - 1], self.U) + self.bh))
            y_pred.append((
                np.dot(h[t], self.V) + self.by).squeeze())
        return y_pred, h

    def predict(self, X, Y, sequenceLength):
        h = []  # stores hidden states
        h.append(np.zeros((1, self.n_h)))  # set initial hidden state to all 0s
        y_pred = []  # stores predictions
        loss = 0

        # produce a prediction for each timestep
        for t in range(sequenceLength):
            h.append(np.tanh(
                np.dot(X[t], self.W) + np.dot(h[t - 1], self.U) + self.bh))
            y_pred.append((
                np.dot(h[t], self.V) + self.by).squeeze())
            loss += (y_pred[t].squeeze() - Y[t].squeeze()) ** 2
        loss = loss/sequenceLength
        print(loss)
        return y_pred, h, loss

    def backprop(self, X, y, y_pred, h):
        # one full iteration of backprop through the time sequence
        dhNext = np.zeros_like(h[0])  # last time step hidden state is all 0s

        for t in reversed(range(self.sequenceLength)):
            dy = y_pred[t] - y[t][0]

            # derivs wrt V matrix
            self.dV += np.dot(h[t].T, dy)
            self.dby += dy

            # derivs wrt inner state
            da = (1 - h[t] ** 2) * \
                (np.dot(dy, self.V.T) + dhNext)
            # derivs wrt prev hidden state
            dhNext = np.dot(da, self.U.T)

            # derivs wrt U, W and b
            self.dU += np.dot(h[t - 1].T, da)
            self.dW += np.dot(X[t].T, da)
            self.dbh += da

        # clip derivs to stop exploding gradient problem
        self.clipDerivs(5, -5)
        return

    def clipDerivs(self, maxClipValue, minClipValue):
        # clips gradients to a max and min value
        # to avoid exploding gradient problem
        if self.dW.min() < minClipValue:
            self.dW[self.dW < minClipValue] = minClipValue
        if self.dU.min() < minClipValue:
            self.dU[self.dU < minClipValue] = minClipValue
        if self.dV.min() < minClipValue:
            self.dV[self.dV < minClipValue] = minClipValue
        if self.dbh.min() < minClipValue:
            self.dbh[self.dbh < minClipValue] = minClipValue
        if self.dby.min() < minClipValue:
            self.dby[self.dby < minClipValue] = minClipValue

        if self.dW.max() > maxClipValue:
            self.dW[self.dW > maxClipValue] = maxClipValue
        if self.dU.max() > maxClipValue:
            self.dU[self.dU > maxClipValue] = maxClipValue
        if self.dV.max() > maxClipValue:
            self.dV[self.dV > maxClipValue] = maxClipValue
        if self.dbh.max() > maxClipValue:
            self.dbh[self.dbh > maxClipValue] = maxClipValue
        if self.dby.max() > maxClipValue:
            self.dby[self.dby > maxClipValue] = maxClipValue
        return

    def learn(self, learningRate):
        # weights
        self.W -= self.dW * learningRate
        self.U -= self.dU * learningRate
        self.V -= self.dV * learningRate

        # biases
        self.bh -= self.dbh * learningRate
        self.by -= self.dby * learningRate
        return

    def train(self, X_batch, y_batch, epochs, learningRate):

        errors = []
        for i in range(epochs):

            y_pred, h = self.forwardPass(X_batch)

            loss = 0
            for t in range(self.sequenceLength):
                loss += (y_pred[t].squeeze() - y_batch[t].squeeze()) ** 2
            errors.append(loss)

            self.backprop(X_batch, y_batch, y_pred, h)
            self.learn(learningRate)
            loss = loss / self.sequenceLength
            print('Epoch:', i + 1, "\t\tLoss:", loss)
        return errors


# preprocess dataset
# function to minmax scale and preprocess input data
def scaleMat(X):
    numer = (X-X.min(axis=0))
    denom = X.max(axis=0) - X.min(axis=0)
    return numer/denom


df = pd.read_csv('ADANIGREEN.NS-indicators.csv')
df.drop('Date', inplace=True, axis=1)
df = df.apply(pd.to_numeric)
X = df.to_numpy()
X = scaleMat(X)
Y = df[["Close"]].to_numpy()
Y = (Y-np.amin(Y)) / \
    (np.amax(Y)-np.amin(Y))
split = 600

x_train = X[:split]
x_train = np.expand_dims(x_train, axis=1)
x_test = X[split: -1]
x_test = np.expand_dims(x_test, axis=1)
Y_train = Y[1: split + 1]
Y_test = Y[split + 1:]
Y_test = np.expand_dims(Y_test, axis=2)
Y_test = np.expand_dims(Y_test, axis=2)


# initialise and train model
model = RNN(hiddenSize=100, sequenceLength=600,
            outputSize=1, inputSize=x_train.shape[2])
errors = model.train(
    x_train, Y_train, epochs=1000, learningRate=0.0001)

predictions, _, loss = model.predict(x_train, Y_train, sequenceLength=600)
plt.title("RNN: Error over Epochs")
plt.plot(errors, label="Error", color="maroon")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend(loc='upper left')
plt.show()

# train dataset
plt.title("RNN: Stock Price Predictions (Train Dataset)")
plt.plot(predictions, label="RNN predictions", color="darkblue")
plt.plot(Y_train.squeeze(), label="True stock value", color="darkorange")
plt.xlabel('Timestep')
plt.ylabel('Stock value (minmax scaled)')
plt.legend(loc='upper left')
plt.show()

# test dataset
predictions, _, loss = model.predict(x_test, Y_test, sequenceLength=171)
plt.title("RNN: Stock Price Predictions (Test Dataset)")
plt.plot(predictions, label="RNN predictions", color="darkblue")
plt.plot(Y_test.squeeze(), label="True stock value", color="darkorange")
plt.xlabel('Timestep')
plt.ylabel('Stock value (minmax scaled)')
plt.legend(loc='upper left')
plt.show()
print("Val error:", loss)
