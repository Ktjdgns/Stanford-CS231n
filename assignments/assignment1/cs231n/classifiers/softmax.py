from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # # Initialize the loss and gradient to zero.
    # loss = 0.0
    # dW = np.zeros_like(W)

    # # compute the loss and the gradient
    # num_classes = W.shape[1]
    # num_train = X.shape[0]
    # for i in range(num_train):
    #     scores = X[i].dot(W)

    #     # compute the probabilities in numerically stable way
    #     scores -= np.max(scores)
    #     p = np.exp(scores)  # Correct answer label is marked as 1.
    #     p /= p.sum()  # normalize
    #     logp = np.log(p)  # Result of softmax
    #     loss -= logp[y[i]]  # negative log probability is the loss 

    #     dW[:, y[i]] = dW[:, y[i]] - X[i]
    #     dW = dW + np.reshape(X[i], (-1,1)).dot(np.reshape(p, (1,-1)))
        
    # # normalized hinge loss plus regularization
    # loss = loss / num_train + reg * np.sum(W * W)
    # dW = dW / num_train + reg * 2 * W

    loss = 0.0
    dW = np.zeros_like(W)

    N = X.shape[0]

    for i in range(N):
        scores = X[i] @ W
        y_exp = np.exp(scores - scores.max())
        softmax = y_exp / y_exp.sum()
        loss -= np.log(softmax[y[i]])
        softmax[y[i]] -= 1
        dW += np.outer(X[i], softmax)

    loss = loss / N + reg * np.sum(W**2)
    dW = dW / N + reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    
    # num_train = X.shape[0]
    # num_class = W.shape[1]

    # scores = X.dot(W)  # (500, 10)
    # scores -= np.reshape(np.max(scores, axis=1), (-1, 1))  # (500, 10)

    # p = np.exp(scores) / np.reshape(np.sum(np.exp(scores), axis=1), (-1, 1))
    
    # result = p[np.arange(num_train), y]
    
    # loss -= np.sum(np.log(result))
    # loss = loss / num_train + reg * np.sum(W * W)

    N = X.shape[0]
    scores = X @ W

    p = np.exp(scores - scores.max())
    p /= p.sum(axis=1, keepdims=True) 

    loss = -np.log(p[range(N), y]).sum()
    loss = loss / N + reg * np.sum(W**2)
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #

    # y_one_hot = np.zeros_like(scores)  # (500, 10)
    # y_one_hot[np.arange(num_train), y] = 1

    # dW = np.transpose(X).dot(np.subtract(scores, y_one_hot))
    # dW = dW / num_train + reg * 2 * W

    p[range(N), y] -= 1
    dW = X.T @ p / N + 2 * reg * W
    #############################################################################


    return loss, dW
