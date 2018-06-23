import numpy as np


class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################
        self.params['w1']=np.random.randn(inputDim,hiddenDim)*0.0001
        self.params['b1']=np.zeros(shape=(1,hiddenDim))
        self.params['w2']=np.random.randn(hiddenDim,outputDim)*0.0001
        self.params['b2']=np.zeros(shape=(1,outputDim))


        pass

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################
        w1,b1=self.params['w1'],self.params['b1']
        w2,b2=self.params['w2'],self.params['b2']


        #forward pass
        z1 = np.dot(x,w1)+b1

        #Laeky Relu activation
        r1=np.maximum(0.01*z1,z1)

        #backward pass
        z2=z1.dot(w2)+b2

        #Leaky Relu activation
        scores=np.maximum(0.01*z2,z2)

        #NN loss
        tmp=-np.log(np.exp(scores[range(len(y)),y])/np.sum(np.exp(scores),axis=1))
        loss=np.sum(tmp)/len(y)+reg*np.sum(w1**2)+reg*np.sum(w2**2)

       #softmax loss
        softmaxprob=np.exp(scores)/np.sum(np.exp(scores),keepdims=True,axis=1)
        dscores=softmaxprob
        dscores[range(len(y)),y]-=1
        dscores/=len(y)
        dscores[scores <= 0] *= 0.01

        #gradient for each parameter

        grads['w2'] = np.dot(z1.T, dscores)
        grads['w2'] += 2*reg * w2
        grads['b2'] = np.sum(dscores, axis=0)

        dz1 = np.dot(dscores, w2.T)
        dz1[z1 <= 0] = 0
        grads['w1'] = np.dot(x.T, dz1)
        grads['w1'] += 2*reg * w1
        grads['b1'] = np.sum(dz1, axis=0)


        pass


        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
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
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################

            numtrain=x.shape[0]
            idx = np.random.choice(numtrain, batchSize, replace=False)
            xBatch=x[idx]
            yBatch=y[idx]
            loss, grads = self.calLoss(x=xBatch, y=yBatch, reg=reg)
            self.params['w1'] -= (lr * grads['w1'])
            self.params['w2'] -= (lr * grads['w2'])
            self.params['b1'] -= (lr * grads['b1'])
            self.params['b2'] -= (lr * grads['b2'])
            lossHistory.append(loss)

            pass

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
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
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        z1 = np.dot(x,w1 + b1)
        a1 = np.maximum(z1,.01 * z1)
        z2 = a1.dot(w2) + b2
        a2 = np.maximum(z2, .01 * z2)
        yPred = np.argmax(a2, axis=1)




        pass



        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        y_pred = self.predict(x)
        acc = (np.mean(y == y_pred)) * 100
        pass



        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



