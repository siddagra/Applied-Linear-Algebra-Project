

## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####
## -------------------------- IMPORTNANT NOTEE !!!:----------------------::####

# I coded my own dot product and transpose function, and it works for simpler
# models like hidden markov model, but since python is a slow language, but when
# I run my more complex neural network models with this, each epoch takes several
# minutes and the whole training takes up to several hours. Neural networks need to
# do thousands of dot products for even one epoch.

# On the contrary, NumPy uses C as its backend which is much faster, and I am able to train 1 epoch
# in a few milliseconds itself. I have included the code for dot product and transpose regardless

# There is nothing wrong with my code and it is not particularly inefficient, it is just that when one
# has to do thousands and millions of dot products, doing it in python takes a lot of time as
# compared to NumPy functions which use C as its backend.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LSTM:
    def __init__(self, inputSize, outputSize, hiddenSize=100, sequenceLength=25,
                 epochs=1000, lr=0.0001):
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.sequenceLength = sequenceLength
        self.epochs = epochs
        self.lr = lr

        self.outputSize = outputSize
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.sequenceLength = sequenceLength
        self.epochs = epochs
        self.lr = lr
        self.concatSize = self.hiddenSize + self.inputSize

        # weights
        self.Wf = np.random.randn(
            self.hiddenSize, self.concatSize) / 1000
        self.Wi = np.random.randn(
            self.hiddenSize, self.concatSize) / 1000
        self.Wc = np.random.randn(
            self.hiddenSize, self.concatSize) / 1000
        self.Wo = np.random.randn(
            self.hiddenSize, self.concatSize) / 1000
        self.Wv = np.random.randn(
            self.outputSize, self.hiddenSize) / 1000

        # biases
        self.bf = np.ones((self.hiddenSize, 1))
        self.bi = np.zeros((self.hiddenSize, 1))
        self.bc = np.zeros((self.hiddenSize, 1))
        self.bo = np.zeros((self.hiddenSize, 1))
        self.bv = np.zeros((self.outputSize, 1))

        # derivatives
        self.dWf = np.zeros((
            self.hiddenSize, self.concatSize))
        self.dWi = np.zeros((
            self.hiddenSize, self.concatSize))
        self.dWc = np.zeros((
            self.hiddenSize, self.concatSize))
        self.dWo = np.zeros((
            self.hiddenSize, self.concatSize))
        self.dWv = np.zeros((
            self.outputSize, self.hiddenSize))

        self.dbf = np.zeros((self.hiddenSize, 1))
        self.dbi = np.zeros((self.hiddenSize, 1))
        self.dbc = np.zeros((self.hiddenSize, 1))
        self.dbo = np.zeros((self.hiddenSize, 1))
        self.dbv = np.zeros((self.outputSize, 1))
        return

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def clipDerivatives(self, maxClipValue, minClipValue):
        # min clipping
        if self.dWf.min() < minClipValue:
            self.dWf[self.dWf < minClipValue] = minClipValue
        if self.dWi.min() < minClipValue:
            self.dWi[self.dWi < minClipValue] = minClipValue
        if self.dWc.min() < minClipValue:
            self.dWc[self.dWc < minClipValue] = minClipValue
        if self.dWo.min() < minClipValue:
            self.dWo[self.dWo < minClipValue] = minClipValue
        if self.dWv.min() < minClipValue:
            self.dWv[self.dWv < minClipValue] = minClipValue

        if self.dbf.min() < minClipValue * 10:
            self.dbf[self.dbf < minClipValue * 10] = minClipValue * 10
        if self.dbi.min() < minClipValue * 10:
            self.dbi[self.dbi < minClipValue * 10] = minClipValue * 10
        if self.dbc.min() < minClipValue * 10:
            self.dbc[self.dbc < minClipValue * 10] = minClipValue * 10
        if self.dbo.min() < minClipValue * 10:
            self.dbo[self.dbo < minClipValue * 10] = minClipValue * 10
        if self.dbv.min() < minClipValue * 10:
            self.dbv[self.dbv < minClipValue * 10] = minClipValue * 10

        # max clipping
        if self.dWf.max() > maxClipValue:
            self.dWf[self.dWf > maxClipValue] = maxClipValue
        if self.dWi.max() > maxClipValue:
            self.dWi[self.dWi > maxClipValue] = maxClipValue
        if self.dWc.max() > maxClipValue:
            self.dWc[self.dWc > maxClipValue] = maxClipValue
        if self.dWo.max() > maxClipValue:
            self.dWo[self.dWo > maxClipValue] = maxClipValue
        if self.dWv.max() > maxClipValue:
            self.dWv[self.dWv > maxClipValue] = maxClipValue

        if self.dbf.max() > maxClipValue * 10:
            self.dbf[self.dbf > maxClipValue * 10] = maxClipValue * 10
        if self.dbi.max() > maxClipValue * 10:
            self.dbi[self.dbi > maxClipValue * 10] = maxClipValue * 10
        if self.dbc.max() > maxClipValue * 10:
            self.dbc[self.dbc > maxClipValue * 10] = maxClipValue * 10
        if self.dbo.max() > maxClipValue * 10:
            self.dbo[self.dbo > maxClipValue * 10] = maxClipValue * 10
        if self.dbv.max() > maxClipValue * 10:
            self.dbv[self.dbv > maxClipValue * 10] = maxClipValue * 10
        return

    def resetDerivatives(self):
        self.dWf.fill(0)
        self.dWi.fill(0)
        self.dWc.fill(0)
        self.dWo.fill(0)
        self.dWv.fill(0)

        self.dbf.fill(0)
        self.dbi.fill(0)
        self.dbc.fill(0)
        self.dbo.fill(0)
        self.dbv.fill(0)
        return

    def learn(self):
        # weights
        self.Wf -= self.lr * self.dWf
        self.Wi -= self.lr * self.dWi
        self.Wc -= self.lr * self.dWc
        self.Wo -= self.lr * self.dWo
        self.Wv -= self.lr * self.dWv

        # biases
        self.bf -= self.lr * self.dbf
        self.bi -= self.lr * self.dbi
        self.bc -= self.lr * self.dbc
        self.bo -= self.lr * self.dbo
        self.bv -= self.lr * self.dbv
        return

    def predict(self, x, y, hPrev, cPrev, sequenceLength):
        # function to predict using LSTM model
        h = hPrev
        c = cPrev
        predictions = []
        loss = 0
        for t in range(sequenceLength):
            yPred, _, h, _, c, _, _, _, _ = self.forwardStep(x[t], h, c)
            predictions.append(yPred[0][0])
        loss = loss / sequenceLength
        return predictions, loss

    def forwardStep(self, x, hPrev, cPrev):
        # forward propogation through one timestep
        z = np.row_stack((hPrev, x))

        # f and i gate
        f = self.sigmoid(np.dot(self.Wf, z) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, z) + self.bi)
        cBar = np.tanh(np.dot(self.Wc, z) + self.bc)

        # output gate
        c = f * cPrev + i * cBar
        o = self.sigmoid(np.dot(self.Wo, z) + self.bo)
        h = o * np.tanh(c)

        # outputs
        yPred = np.dot(self.Wv, h) + self.bv
        return yPred, yPred, h, o, c, cBar, i, f, z

    def backwardStep(self, y, yPred, dhNext, dcNext, cNext, z, f, i, cBar, c, o, h):
        # backpropogation through one timestep
        # derivative of error with respect to output
        dv = np.copy(yPred)
        dv[0][0] = yPred[0][0] - y

        self.dWv += np.dot(dv, h.T)
        self.dbv += dv

        dh = np.dot(self.Wv.T, dv)
        dh += dhNext

        # derivatives with respect to output weight and biases
        do = dh * np.tanh(c) * o * (1 - o)
        self.dWo += np.dot(do, z.T)
        self.dbo += do

        dc = dh * o * (1 - np.tanh(c) ** 2)
        dc += dcNext

        # derivatives with respect to outputgate weight and biases
        da = dc * i * (1 - cBar ** 2)
        self.dWc += np.dot(da, z.T)
        self.dbc += da

        # derivatives with respect to input weight and biases
        di = dc * cBar * i * (1 - i)
        self.dWi += np.dot(di, z.T)
        self.dbi += di

        # derivatives with respect to forget weight and biases
        df = dc * cNext * f * (1 - f)
        self.dWf += np.dot(df, z.T)
        self.dbf += df

        # derivative with respect to last output
        dz = (np.dot(self.Wf.T, df)
              + np.dot(self.Wi.T, di)
              + np.dot(self.Wc.T, dc)
              + np.dot(self.Wo.T, do))

        dhNext = dz[:self.hiddenSize, :]
        dcNext = f * dc
        return dhNext, dcNext

    def oneIter(self, X, Y, hPrev, cPrev):
        # one iteration of both forward and backward propogations
        x, z, f, i, cBar, c, o, yPred, v, h = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        h[-1] = hPrev
        c[-1] = cPrev

        loss = 0
        for t in range(self.sequenceLength):
            x[t] = X[t]
            yPred[t], v[t], h[t], o[t], c[t], cBar[t], i[t], f[t], z[t] = self.forwardStep(
                x[t], h[t - 1], c[t - 1])
            loss += (yPred[t] - Y[t]) ** 2
        self.resetDerivatives()

        dh   = np.zeros_like(h[0])
        dcNext = np.zeros_like(c[0])

        for t in reversed(range(self.sequenceLength)):
            dhNext, dcNext = self.backwardStep(Y[t], yPred[t], dhNext,
                                               dcNext, c[t -
                                                         1], z[t], f[t], i[t],
                                               cBar[t], c[t], o[t], h[t])
        loss = loss/self.sequenceLength
        return loss, h[self.sequenceLength - 1], c[self.sequenceLength - 1]

    def train(self, X, Y):
        errors = []

        for epoch in range(self.epochs):
            hPrev = np.zeros((self.hiddenSize, 1))
            cPrev = np.zeros((self.hiddenSize, 1))

            loss, hPrev, cPrev = self.oneIter(
                X, Y, hPrev, cPrev)

            errors.append(loss.squeeze())
            self.learn()
            self.clipDerivatives(5, -5)

            print('Epoch:', epoch, '\tLoss:', loss)
        return errors


# preprocess dataset
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

x_train = X[: split]
x_train = np.expand_dims(x_train, axis=2)
x_test = X[split: -1]
x_test = np.expand_dims(x_test, axis=2)


Y_train = Y[1: split + 1]
Y_test = Y[split + 1:]
Y_train = Y_train.squeeze()
Y_test = Y_test.squeeze()


# initialise and train model
model = LSTM(inputSize=89, hiddenSize=100,
             outputSize=1, epochs=1000, lr=0.001, sequenceLength=600)
errors = model.train(x_train, Y_train)


# train dataset
predictions, loss = model.predict(x=x_train, y=Y_train, hPrev=np.zeros(
    (100, 1)), cPrev=np.zeros((100, 1)), sequenceLength=600)


plt.title("LSTM: Error over Epochs")
plt.plot(errors, label="Error", color="maroon")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend(loc='upper left')
plt.show()

plt.title("LSTM: Stock Price Predictions (Train Dataset)")
plt.plot(predictions, label="LSTM predictions", color="darkblue")
plt.xlabel('Timestep')
plt.ylabel('Stock value')
plt.legend(loc='upper left')
plt.show()

plt.title("LSTM: Stock True Value (Train Dataset)")
plt.plot(Y_train.squeeze(), label="True stock value", color="darkorange")
plt.xlabel('Timestep')
plt.ylabel('Stock value')
plt.legend(loc='upper left')
plt.show()


# test dataset
predictions, loss = model.predict(x=x_test, y=Y_test, hPrev=np.zeros(
    (100, 1)), cPrev=np.zeros((100, 1)), sequenceLength=171)
plt.title("LSTM: Stock Price Predictions (Test Dataset)")
plt.plot(predictions, label="LSTM predictions", color="darkblue")
plt.xlabel('Timestep')
plt.ylabel('Stock value (minmax scaled)')
plt.legend(loc='upper left')
plt.show()


plt.title("LSTM: Stock Price Predictions (Test Dataset)")
plt.plot(Y_test.squeeze(), label="True stock value", color="darkorange")
plt.xlabel('Timestep')
plt.ylabel('Stock value (minmax scaled)')
plt.legend(loc='upper left')
plt.show()
