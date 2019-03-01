import numpy as np


class Softmax (object):
    """" Softmax classifier """

    def __init__ (self, inputDim, outputDim):
        self.W = None
        #########################################################################
        # TODO: 5 points                                                        #
        # - Generate a random softmax weight matrix to use to compute loss.     #
        #   with standard normal distribution and Standard deviation = 0.01.    #
        #########################################################################
        self.W = np.random.random((inputDim, outputDim)) * 0.01

        pass
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        Softmax loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to weights self.W (dW) with the same shape of self.W.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)
        #############################################################################
        # TODO: 20 points                                                           #
        # - Compute the softmax loss and store to loss variable.                    #
        # - Compute gradient and store to dW variable.                              #
        # - Use L2 regularization                                                  #
        # Bonus:                                                                    #
        # - +2 points if done without loop                                          #
        #############################################################################

        S = x.dot(self.W)
        S1 = S - np.max(S,axis =1, keepdims = True)
        exp_S1 = np.exp(S1)
        sum_f = np.sum(exp_S1,axis = 1, keepdims = True)
        P_yi = exp_S1[np.arange(x.shape[0]),y]/sum_f
        loss_i = -np.log(P_yi)

        loss = np.sum(loss_i)/x.shape[0]
        loss = loss + reg * np.sum(self.W * self.W)

        dS = exp_S1/sum_f
        dS[range(x.shape[0]), y] -= 1
        dS = dS/x.shape[0]

        dW = np.dot(x.T, dS)
        dW = dW + 2* reg * self.W


        pass
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Softmax classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (D, batchSize)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################

            batch = np.random.choice(np.arange(x.shape[0]), batchSize, replace=False)
            xBatch = x[batch]
            yBatch = y[batch]

            loss, dW = self.calLoss(x[batch], y[batch], reg)

            lossHistory.append(loss)

            self.W = self.W - lr * dW

            pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Store the predict output in yPred                                    #
        ###########################################################################

        Score = x.dot(self.W)
        yPred = np.argmax(Score,axis = 1)

        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################

        acc = np.mean(y == self.predict(x))

        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



