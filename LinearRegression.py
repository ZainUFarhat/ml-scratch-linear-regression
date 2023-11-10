# utils
from utils import *

# Linear Regression
class LinearRegression():

    """
    Description:
        My from scratch implementation of the Linear Regression Algorithm
    """

    # constructor
    def __init__(self, epochs, lr):

        """
        Description:
            Contructor for our LinearRegression class
        
        Parameters:
            epochs: number of training iterations
        
        Returns:
            None
        """

        # epochs and learning rate
        self.epochs = epochs
        self.lr = lr
        # weights and biases
        self.w = []
        self.b = 0
    
    # fit
    # this is the training process with a simple gradient descent for optimization
    def fit(self, X, y):

        """
        Description:
            Fits the training data of our Linear Regression model

        Parameters:
            X: predictors
            y: responses
        
        Returns:
            mse_losses
        """

        # extract number of training samples N and number of features
        N, num_features = X.shape

        # intialize weights to zero
        self.w = np.zeros(num_features)

        # list to append mse loss per iteration
        mse_losses = []
        
        # iterate through our training set epochs times
        # note: we are pursuing a vectorized approach to update and optimize weights and biases
        # there will be no nested for loops and any indexing
        for _ in range(self.epochs):

            # predict with current state of weights and bias
            y_pred = np.dot(X, self.w) + self.b

            # find the loss at current iteration and append it to list
            mse_loss = mse(y, y_pred)
            mse_losses.append(mse_loss)

            # compute the first derivatives of mse loss with respect to weights and bias respectively
            dw = (1/N) * 2 * np.dot(X.T, (y_pred - y))
            db = (1/N) * 2 * np.sum(y_pred - y)

            # update weights and biases
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
        
        # return
        return mse_losses
    
    # predict
    def predict(self, X):

        """
        Description:
            Predicts based on the fitted Linear Regression model
        
        Parameters:
            X: unseen data we want to predict on
        
        Returns:
            y_pred
        """

        # predict on unseen data with our optimized weights and bias
        y_pred = np.dot(X, self.w) + self.b

        # return
        return y_pred