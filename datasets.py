# pandas
import pandas as pd

# sklearn
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

class Datasets():

    """
    Description:
        Holds different regression datasets
    """

    def __init__(self):

        pass

    def simple_Xy(self, n_samples, n_features, noise, test_size, random_state):
        
        """
        Description:
            Simple X vs. y dataset created by sklearn
        
        Parameters:
            n_samples: number of samples to genarate
            n_features: number of features (corresponds to number of dimension)
            noise: standard deviation of the gaussian noise applied to the output
            test_size: percentage of data to be allocated for testing
            random_state: random state chosen for reproducible output
        
        Returns:
            X, y, column_names, X_train, X_test, y_train, y_test
        """

        # create dataset
        X, y = datasets.make_regression(n_samples = n_samples, n_features = n_features, noise = noise, random_state = random_state)

        # set column names
        column_names = ['X', 'y']

        # normalize both X & y to range = [0, 1] to avoid explosive computations
        X = minmax_scale(X, feature_range = (0, 1), axis = 0, copy = True)
        y = minmax_scale(y, feature_range = (0, 1), axis = 0, copy = True)

        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

        # return
        return X, y, column_names, X_train, X_test, y_train, y_test

    
    def tv_marketing(self, test_size, random_state):

        """
        Description:
            loads and splits the TV Marketing dataset
        
        Parameters:
            test_size: percentage of data to be allocated for testing
            random_state: random state chosen for reproducible output
        
        Returns:
            X, y, column_names, X_train, X_test, y_train, y_test
        """

        # load data
        data = pd.read_csv('data/tvmarketing.csv')

        column_names = list(data.columns)

        # assign predictor and response variables
        X = data['TV'].values
        y = data['Sales'].values

        # add an extra dimension to account for the one feature
        X = X.reshape(-1,1)

        # normalize both X & y to range = [0, 1] to avoid explosive computations
        X = minmax_scale(X, feature_range = (0, 1), axis = 0, copy = True)
        y = minmax_scale(y, feature_range = (0, 1), axis = 0, copy = True)

        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

        # return
        return X, y, column_names, X_train, X_test, y_train, y_test

    def bmi_life_expectancy(self, test_size, random_state):

        """
        Description:
            loads and splits the BMI and Life Expectancy Dataset
        
        Parameters:
            test_size: percentage of data to be allocated for testing
            random_state: random state chosen for reproducible output
        
        Returns:
            X, y, column_names, X_train, X_test, y_train, y_test  
        """

        # load data
        data = pd.read_csv('data/bmi_and_life_expectancy.csv')

        # fetch column names
        column_names = list(data.columns[1:])

        # assign predictor and response variables
        X = data['Life expectancy'].values
        y = data['BMI'].values

        # add an extra dimension to account for the one feature
        X = X.reshape(-1,1)

        # normalize both X & y to range = [0, 1] to avoid explosive computations
        X = minmax_scale(X, feature_range = (0, 1), axis = 0, copy = True)
        y = minmax_scale(y, feature_range = (0, 1), axis = 0, copy = True)

        # perform train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

        # return
        return X, y, column_names, X_train, X_test, y_train, y_test