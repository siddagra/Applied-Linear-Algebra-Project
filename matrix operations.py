import numpy as np


def dot(x, y):
    assert x.shape[1] == y.shape[0], "Incorrect matrix dimensions for dot product. The number of columns of the 1st matrix must equal to the number of rows of the 2nd matrix."
    dotProduct = np.zeros((x.shape[0], y.shape[1]))
    for i in range(0, x.shape[0]):
        for j in range(0, y.shape[1]):
            for k in range(0, x.shape[1]):
                dotProduct[i][j] += x[i][k] * y[k][j]
    return dotProduct


def transpose(x):
    transposedMat = np.zeros((x.shape[1], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            transposedMat[j][i] = x[i][j]
    return transposedMat


def hadamard(x, y):
    assert x.shape[1] == y.shape[1] and x.shape[0] == y.shape[0], "Incorrect matrix dimensions for hadamard product. The dimensions of both matrices must be equal for an elementwise product."
    hadamardProduct = np.zeros((x.shape[1], x.shape[0]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            hadamardProduct[i][j] = x[i][j] * y[i][j]
    return hadamardProduct


def outer(x, y):
    return dot(transpose(x), y)
