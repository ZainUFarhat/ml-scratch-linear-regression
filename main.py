# datasets
from datasets import *

# Linear Regression
from LinearRegression import *

# set numpy random seed
np.random.seed(42)

# Xy
def main_xy():

    """
    Description:
        Main function to train Linear Regression Model on the Simple Xy Dataset
    
    Parameters:
        None

    Returns:
        None    
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    test_size = 0.3
    random_state = 42
    n_samples = 100
    n_features = 1
    noise = 20
    dataset_name = 'Simple Xy' 

    # create instance of Datasets class
    xy = Datasets()
    # load the tv marketing class
    X, y, column_names, X_train, X_test, y_train, y_test = xy.simple_Xy(n_samples = n_samples, n_features = n_features, noise = noise, 
                                                                            test_size = test_size, random_state = random_state)

    print('Loading Simple Xy Dataset...')
    print('\nThe Predictor and Response Variables of the Simple Xy Dataset are:', ', '.join(column_names), 'respectively')
    print(f'\nSimple Xy contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nLinear Regression\n')

    print('---------------------------------------------------Training---------------------------------------------------')
    # regression hyperparameters
    epochs = 1000
    lr = 0.1

    print('Number of epochs =', epochs)
    print('Learning rate =', lr)
    print('Distance metric used is the Mean Squared Error (MSE)')
    print('Training in progress...')

    # create an instance of linear regression class
    lreg = LinearRegression(epochs = epochs, lr = lr)
    # fit it (train)
    mse_train_losses = lreg.fit(X = X_train, y = y_train)
    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    # predict
    predictions = lreg.predict(X_test)
    # get test mse
    mse_test_final = mse(y = y_test, y_pred = predictions)
    print('Test MSE =', round(mse_test_final, 4)) 
    print('---------------------------------------------------Plotting---------------------------------------------------')
    # scatter plot of original data
    title_scatter = f'{dataset_name} - {column_names[0]} vs. {column_names[1]}'
    save_path_scatter = 'plots/xy/xy_scatter.png'
    scatter_plot(X = X, y = y, title = title_scatter, x_label = column_names[0], y_label = column_names[1], 
                                savepath = save_path_scatter, fitted_line = False, y_preds = None)
    # training curve
    title_loss = f'{dataset_name} - MSE Training Loss'
    save_path_loss = 'plots/xy/xy_loss.png'
    plot_training_curve(epochs = list(range(1, epochs+1)), losses = mse_train_losses, title = title_loss,
                                                                     x_label = 'epochs', y_label = 'mse', savepath = save_path_loss)
    # plot linear regression line
    title_fitted = f'{dataset_name} - Fitted Line'
    save_path_fitted = 'plots/xy/xy_fitted.png' 
    # get predictions on entire dataset
    y_preds = lreg.predict(X)
    scatter_plot(X = X, y = y, title = title_fitted, x_label = column_names[0], y_label = column_names[1], 
                                savepath = save_path_fitted, fitted_line = True, y_preds = y_preds)
    
    # plot the alignment between true and predicted responses
    save_path_alignment = 'plots/xy/xy_alignment.png'
    idx_range = list(range(1, len(y_test)+1, 1))
    plot_actual_predicted_index(idx_range = idx_range, y = y_test, y_pred = predictions, y_label = column_names[1],
                                                                    dataset_name = dataset_name, savepath = save_path_alignment)
    print('Please refer to plots/xy directory to view all plots.')
    print('--------------------------------------------------------------------------------------------------------------')

    # return
    return None

# tv
def main_tv():

    """
    Description:
        Main function to train Linear Regression Model on the TV Marketing Dataset
    
    Parameters:
        None

    Returns:
        None    
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    test_size = 0.3
    random_state = 42
    dataset_name = 'TV Marketing'

    # create instance of Datasets class
    tv = Datasets()
    # load the tv marketing class
    X, y, column_names, X_train, X_test, y_train, y_test = tv.tv_marketing(test_size = test_size, random_state = random_state)

    print('Loading TV Marketing Dataset...')
    print('\nThe Predictor and Response Variables of the TV Marketing Dataset are:', ', '.join(column_names), 'respectively')
    print(f'\nTV Marketing contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nLinear Regression\n')

    print('---------------------------------------------------Training---------------------------------------------------')
    # regression hyperparameters
    epochs = 100
    lr = 0.4

    print('Number of epochs =', epochs)
    print('Learning rate =', lr)
    print('Distance metric used is the Mean Squared Error (MSE)')
    print('Training in progress...')

    # create an instance of linear regression class
    lreg = LinearRegression(epochs = epochs, lr = lr)
    # fit it (train)
    mse_train_losses = lreg.fit(X = X_train, y = y_train)
    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    # predict
    predictions = lreg.predict(X_test)
    # get test mse
    mse_test_final = mse(y = y_test, y_pred = predictions)
    print('Test MSE =', round(mse_test_final, 4)) 
    print('---------------------------------------------------Plotting---------------------------------------------------')
    # scatter plot of original data
    title_scatter = f'{dataset_name} - {column_names[0]} vs. {column_names[1]}'
    save_path_scatter = 'plots/tv/tv_scatter.png'
    scatter_plot(X = X, y = y, title = title_scatter, x_label = column_names[0], y_label = column_names[1], 
                                savepath = save_path_scatter, fitted_line = False, y_preds = None)
    # training curve
    title_loss = f'{dataset_name} - MSE Training Loss'
    save_path_loss = 'plots/tv/tv_loss.png'
    plot_training_curve(epochs = list(range(1, epochs+1)), losses = mse_train_losses, title = title_loss,
                                                                     x_label = 'epochs', y_label = 'mse', savepath = save_path_loss)
    # plot linear regression line
    title_fitted = f'{dataset_name} - Fitted Line'
    save_path_fitted = 'plots/tv/tv_fitted.png' 
    # get predictions on entire dataset
    y_preds = lreg.predict(X)
    scatter_plot(X = X, y = y, title = title_fitted, x_label = column_names[0], y_label = column_names[1], 
                                savepath = save_path_fitted, fitted_line = True, y_preds = y_preds)
    
    # plot the alignment between true and predicted responses
    save_path_alignment = 'plots/tv/tv_alignment.png'
    idx_range = list(range(1, len(y_test)+1, 1))
    plot_actual_predicted_index(idx_range = idx_range, y = y_test, y_pred = predictions, y_label = column_names[1],
                                                                        dataset_name = dataset_name, savepath = save_path_alignment)
    print('Please refer to plots/tv directory to view all plots.')
    print('--------------------------------------------------------------------------------------------------------------') 
    
    # return
    return None

# bmi
def main_bmi():

    """
    Description:
        Main function to train Linear Regression Model on the BMI and Life Expectancy Dataset
    
    Parameters:
        None

    Returns:
        None    
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    test_size = 0.3
    random_state = 42
    dataset_name = 'BMI & Life Expectancy'

    # create instance of Datasets class
    bmi = Datasets()
    # load the tv marketing class
    X, y, column_names, X_train, X_test, y_train, y_test = bmi.bmi_life_expectancy(test_size = test_size, random_state = random_state)

    print('Loading BMI and Life Expectancy Dataset...')
    print('\nThe Predictor and Response Variables of the BMI and Life Expectancy Dataset are:', ', '.join(column_names), 'respectively')
    print(f'\nBMI and Life Expectancy contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nLinear Regression\n')

    print('---------------------------------------------------Training---------------------------------------------------')
    # regression hyperparameters
    epochs = 1000
    lr = 0.1

    print('Number of epochs =', epochs)
    print('Learning rate =', lr)
    print('Distance metric used is the Mean Squared Error (MSE)')
    print('Training in progress...')

    # create an instance of linear regression class
    lreg = LinearRegression(epochs = epochs, lr = lr)
    # fit it (train)
    mse_train_losses = lreg.fit(X = X_train, y = y_train)
    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    # predict
    predictions = lreg.predict(X_test)
    # get test mse
    mse_test_final = mse(y = y_test, y_pred = predictions)
    print('Test MSE =', round(mse_test_final, 4)) 
    print('---------------------------------------------------Plotting---------------------------------------------------')
    # scatter plot of original data
    title_scatter = f'{dataset_name} - {column_names[0]} vs. {column_names[1]}'
    save_path_scatter = 'plots/bmi/bmi_scatter.png'
    scatter_plot(X = X, y = y, title = title_scatter, x_label = column_names[0], y_label = column_names[1], 
                                savepath = save_path_scatter, fitted_line = False, y_preds = None)
    # training curve
    title_loss = f'{dataset_name} - MSE Training Loss'
    save_path_loss = 'plots/bmi/bmi_loss.png'
    plot_training_curve(epochs = list(range(1, epochs+1)), losses = mse_train_losses, title = title_loss,
                                                                     x_label = 'epochs', y_label = 'mse', savepath = save_path_loss)
    # plot linear regression line
    title_fitted = f'{dataset_name} - Fitted Line'
    save_path_fitted = 'plots/bmi/bmi_fitted.png' 
    # get predictions on entire dataset
    y_preds = lreg.predict(X)
    scatter_plot(X = X, y = y, title = title_fitted, x_label = column_names[0], y_label = column_names[1], 
                                savepath = save_path_fitted, fitted_line = True, y_preds = y_preds)
    
    # plot the alignment between true and predicted responses
    save_path_alignment = 'plots/bmi/bmi_alignment.png'
    idx_range = list(range(1, len(y_test)+1, 1))
    plot_actual_predicted_index(idx_range = idx_range, y = y_test, y_pred = predictions, y_label = column_names[1], 
                                                                            dataset_name = dataset_name, savepath = save_path_alignment)
    print('Please refer to plots/bmi directory to view all plots.')
    print('--------------------------------------------------------------------------------------------------------------') 
    
    # return
    return None

if __name__ == '__main__':

    main_xy()
    main_tv()
    main_bmi()