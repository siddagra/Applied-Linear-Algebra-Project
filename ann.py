import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# class to initialise and store weights for a single forward propogation step
# often called as a dense layer in AI/ML nomenclature
class Dense:
    def __init__(self, dim0, dim1, prevOut):
        self.prevOut = prevOut  # previous output required to compute next output

        # initialise weights and biases of the layer
        self.weights = np.random.randn(dim0, dim1)
        self.bias = np.random.randn(1, 1)

        self.out = None

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x*(1-x)
        return 1/(1+np.exp(-x))

    def compute_output(self, prevOut):
        self.prevOut = prevOut
        self.out = self.sigmoid(np.dot(
            self.prevOut, self.weights))


# class to store the whole ANN network
class ANN:
    def __init__(self, X, Y):
        self.X = X
        self.HiddenLayers = []
        self.Y = Y

    # mean squared error function
    def mse(self, y_pred, y):
        return 0.5*(y_pred-y)**2

    # derivative of mse for backprop
    def mseDeriv(self, y_pred, y):
        return y_pred*(y_pred-y)

    # sigmoid function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # derivative of sigmoid for backprop
    def sigmoidDeriv(self, x):
        return x*(1-x)

    def addHiddenLayer(self, hiddenSize=5):
        # if we have no existing hidden HiddenLayers,
        # then map from input to hidden HiddenLayer
        if len(self.HiddenLayers) == 0:
            HiddenLayer = Dense(self.X.shape[1], hiddenSize, prevOut=self.X)
        else:
            # otherwise just map from hidden HiddenLayer
            # to another hidden HiddenLayer or output
            HiddenLayer = Dense(
                self.HiddenLayers[-1].weights.shape[1], hiddenSize,
                prevOut=self.HiddenLayers[-1].out)
        # append hidden HiddenLayer to ANN model
        self.HiddenLayers.append(HiddenLayer)

    def forwardPass(self):
        # one iteration of forward propogation
        # computes outputs

        prevOut = self.HiddenLayers[0].prevOut
        for HiddenLayer in self.HiddenLayers:
            HiddenLayer.compute_output(prevOut)
            prevOut = HiddenLayer.out

    def backPropogate(self, learningRate):
        # one iteration of backpropogation
        # calculates the derivatives, and updates weights
        # and bias according to the learning rate

        temp = outputLayer = self.HiddenLayers[-1]

        # compute derivatives of error/loss
        # with respect to output layer
        deriv = self.mseDeriv(outputLayer.out, self.Y) * \
            self.sigmoidDeriv(outputLayer.out)

        # update weights
        outputLayer.weights -= learningRate * \
            np.dot(outputLayer.prevOut.T, deriv)

        # update bias
        outputLayer.bias -= learningRate * np.mean(deriv)

        # np.flip used to reverse the hidden layers
        # so that we can backpropogate through them
        for HiddenLayer in np.flip(self.HiddenLayers, axis=0)[1:]:

            # compute derivatives of error/loss
            # with respect to previous hidden layer
            deriv = np.dot(deriv, temp.weights.T) * \
                self.sigmoidDeriv(HiddenLayer.out)

            # update weights
            HiddenLayer.weights -= learningRate * \
                np.dot(HiddenLayer.prevOut.T, (deriv))

            # update bias
            HiddenLayer.bias -= learningRate * np.mean(deriv)

            # store hidden layer n+1 as a temporary variable so that
            # we can compute derivatives for the next layer
            temp = HiddenLayer

    def train(self, epochs=65000, learningRate=0.005):
        self.HiddenLayers = np.array(self.HiddenLayers)
        errors = []
        for i in range(epochs):
            self.forwardPass()
            self.backPropogate(learningRate)
            error = np.mean(
                self.mse(self.HiddenLayers[-1].out, self.Y))
            errors.append(error)
            if i % 1000 == 0:
                print("mse train:", error)
        return errors

    def predict(self, X):
        prevOut = np.array(X)
        for HiddenLayer in self.HiddenLayers:
            HiddenLayer.compute_output(prevOut)
            prevOut = HiddenLayer.out
        return prevOut


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
x_test = X[split: -1]


Y_train = Y[1: split + 1].squeeze()
Y_test = Y[split + 1:].squeeze()

Y_train = Y_train
Y_test = Y_test

Y_train = np.expand_dims(Y_train, axis=1)

# initialise and train model
ann = ANN(x_train, Y_train)
ann.addHiddenLayer(5)
ann.addHiddenLayer(1)
error = ann.train(epochs=50000, learningRate=0.005)


# train dataset
predictions_train = ann.predict(x_train)
predictions_test = ann.predict(x_test)

plt.title("Error over epochs")
plt.plot(error, label="Error")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend(loc='upper left')
plt.show()

plt.title("ANN: Predictions vs. True Stock Value over Time (Train Dataset)")
plt.plot(predictions_train, label="ANN predictions")
plt.plot(Y_train.squeeze(), label="True stock value")
plt.xlabel('Timestep')
plt.ylabel('Stock value (minmax scaled)')
plt.legend(loc='upper left')
plt.show()


# test dataset
plt.title("ANN: Predictions vs. True Stock Value over Time (Test Dataset)")
plt.plot(predictions_test, label="ANN predictions")
plt.plot(Y_test.squeeze(), label="True stock value")
plt.xlabel('Timestep')
plt.ylabel('Stock value (minmax scaled)')
plt.legend(loc='upper left')
plt.show()
