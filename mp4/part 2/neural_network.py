import numpy as np

eta = 0.1
"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):

    # Batch size, number of examples, and parameter restructuring
    n = 200
    N = x_train.shape[0]
    Ws = [w1, w2, w3, w4]
    bs = [b1, b2, b3, b4]
    losses = []

    # Iterating through our epochs
    for e in range(epoch):

        # Shuffling our data if necessary
        y_train_shuff = y_train
        x_train_shuff = x_train
        if shuffle:
            y_train_temp = y_train.reshape((1, y_train.shape[0]))
            temp = np.concatenate((y_train_temp.T, x_train), axis=1)
            np.random.shuffle(temp)

            y_train_shuff = temp[:, 0]
            x_train_shuff = temp[:, 1:]

        # Splitting our data up into batches
        total_loss = 0
        for i in range(N // n):
            start_idx = i * n
            end_idx = min((i + 1) * n, N)

            x_batch = x_train_shuff[start_idx:end_idx]
            y_batch = y_train_shuff[start_idx:end_idx]
            total_loss += four_nn(x_batch, Ws, bs, y_batch, test=False)

        losses.append(total_loss)

    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):

    Ws = [w1, w2, w3, w4]
    bs = [b1, b2, b3, b4]
    classifications = four_nn(x_test, Ws, bs, None, test=True)

    correct_class = np.where(classifications == y_test, 1, 0)

    avg_class_rate = np.sum(correct_class) / x_test.shape[0]
    class_rate_per_class = [0.0] * num_classes
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(X, Ws, bs, y, test):
    Z1, acache1 = affine_forward(X, Ws[0], bs[0])
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1, Ws[1], bs[1])
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2, Ws[2], bs[2])
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3, Ws[3], bs[3])

    if test:
        classification = np.argmax(F, axis=1)
        return classification

    loss, dF = cross_entropy(F, y)
    dA3, dW4, db4 = affine_backward(dF, acache4)
    dZ3 = relu_backward(dA3, rcache3)
    dA2, dW3, db3 = affine_backward(dZ3, acache3)
    dZ2 = relu_backward(dA2, rcache2)
    dA1, dW2, db2 = affine_backward(dZ2, acache2)
    dZ1 = relu_backward(dA1, rcache1)
    dX, dW1, db1 = affine_backward(dZ1, acache1)

    # Update parameters
    Ws[0] -= eta * dW1
    Ws[1] -= eta * dW2
    Ws[2] -= eta * dW3
    Ws[3] -= eta * dW4

    bs[0] -= eta * db1
    bs[1] -= eta * db2
    bs[2] -= eta * db3
    bs[3] -= eta * db4

    return loss

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    z_rows = A.shape[0]
    z_cols = W.shape[1]
    Z = np.zeros((z_rows, z_cols))
    d = W.shape[0]

    Z = A @ W + b

    cache = [A, W]
    return Z, cache

def affine_backward(dZ, cache):
    A = cache[0]
    W = cache[1]

    dA = np.zeros(A.shape)
    dW = np.zeros(W.shape)
    db = np.zeros((W.shape[1]))

    dA = dZ @ W.T
    dW = A.T @ dZ
    dB = np.sum(dZ, axis=0)

    return dA, dW, dB

def relu_forward(Z):
    A = np.maximum(Z, 0)

    cache = [A]
    return A, cache

def relu_backward(dA, cache):
    Z = cache[0]
    dZ = np.where(Z <= 0, 0, dA)

    return dZ

def cross_entropy(F, y):
    n = F.shape[0]
    C = F.shape[1]

    # Computing loss
    loss = 0
    for i in range(n):
        loss_sum = np.exp(F[i]).sum()
        loss += F[i][int(y[i])] - np.log(loss_sum)

    loss *= -(1/n)

    # Computing dF
    binary = np.zeros(F.shape)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if j == y[i]:
                binary[i][j] = 1

    boardcasted = np.zeros(F.shape) + np.sum(np.exp(F), axis=1).reshape((F.shape[0], 1))
    dF = binary - np.exp(F) / boardcasted
    dF *= -(1/n)

    return loss, dF
