import numpy as np

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

    #IMPLEMENT HERE

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

    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn():
    pass

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
# TODO: replace naive python loops with numpy multiplication
def affine_forward(A, W, b):
    z_rows = A.shape[0]
    z_cols = W.shape[1]
    Z = np.zeros((z_rows, z_cols))
    d = W.shape[0]

    for i in range(z_rows):
        for j in range(z_cols):
            z_ij = 0
            for k in range(d):
                z_ij += A[i][k] * W[k][j]
            Z[i][j] = z_ij + b[j]

    cache = [A, W]
    return Z, cache

# TODO: replace naive python loops with numpy multiplication
def affine_backward(dZ, cache):
    A = cache[0]
    W = cache[1]

    dA = np.zeros(A.shape)
    dW = np.zeros(W.shape)
    db = np.zeros((W.shape[1]))

    n = A.shape[0]
    d_prime = W.shape[1]

    # Computing dA
    for i in range(dA.shape[0]):
        for k in range(dA.shape[1]):
            dA_ik = 0
            for j in range(d_prime):
                dA_ik += dZ[i][j] * W[k][j]
            dA[i][k] = dA_ik

    # Computing dW
    for k in range(dW.shape[0]):
        for j in range(dW.shape[1]):
            dW_kj = 0
            for i in range(n):
                dW_kj += A[i][k] * dZ[i][j]
            dW[k][j] = dW_kj

    # Computing db
    for j in range(db.shape[0]):
        db_j = 0
        for i in range(n):
            db_j += dZ[i][j]
        db[j] = db_j

    return dA, dW, db

def relu_forward(Z):
    A = np.maximum(Z, 0)

    cache = [A]
    return A, cache

def relu_backward(dA, cache):
    Z = cache[0]
    dZ = np.where(Z <= 0, 0, dA)

    return dZ

# TODO: replace naive python loops with numpy multiplication
def cross_entropy(F, y):
    n = F.shape[0]
    C = F.shape[1]

    # Computing loss
    loss = 0
    for i in range(n):
        loss += F[i][int(y[i])]

        loss_sum = 0
        for k in range(C):
            loss_sum += np.exp(F[i][k])
        loss -= np.log(loss_sum)

    loss *= -(1/n)

    # Computing dF
    dF = np.zeros(F.shape)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            dF_ij = 0

            if j == y[i]:
                dF_ij += 1

            dF_sum = 0
            for k in range(C):
                dF_sum += np.exp(F[i][k])

            dF_ij -= np.exp(F[i][j]) / dF_sum
            dF_ij *= -(1/n)
            dF[i][j] = dF_ij

    return loss, dF
