# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# mean squared error loss
def mse(y, y_pred):

    """
    Description:
        Calculates the mean squared error loss

    Parameters:
        y: original responses
        y_pred: predicted responses
    
    Returns:
        mse_loss
    """

    mse_loss = np.mean((y - y_pred) ** 2)

    return mse_loss

# scatter plot of given data
def scatter_plot(X, y, title, x_label, y_label, savepath, fitted_line, y_preds):

    """
    Description:
        Plots a scatterplot based on X & y data provided

    Parameters:
        X: x-axis datapoints
        y: y-axis datapoints
        title: tite of plot
        x_label: label for x axis
        y_label: label for y axis
        savepath: path to save our scatterplot to
        fitted_line: boolean variable that decides whether to print a fitted line or not
        y_preds: predictions against original data

    Returns:
        None
    """

    plt.figure(figsize = (7, 7))

    ax = plt.axes()
    ax.set_facecolor("lavender")

    plt.scatter(X, y, c = 'g')
    if fitted_line == True:
            plt.plot(X, y_preds, c = 'r')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(savepath)

    # return
    return None

# plot training curve
def plot_training_curve(epochs, losses, title, x_label, y_label, savepath):

    """
    Description:
        Plots the loss per epoch during fitting our Linear Regressor
    
    Parameters:
        epochs: list holding range of number of epochs trained on
        losses: our training losses
        title: tite of plot
        x_label: label for x axis
        y_label: label for y axis
        savepath: path to save our scatterplot to

    Returns:
        None
    """

    plt.figure(figsize = (7, 7))

    ax = plt.axes()
    ax.set_facecolor("lavender")

    plt.plot(epochs, losses, c = 'b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(savepath)

    # return
    return None

# Actual vs Predicted
def plot_actual_predicted_index(idx_range, y, y_pred, y_label, dataset_name, savepath):

    """
    Description:
        Plots the alignment between the true and predicted responses
    
    Parameters:
        idx_range: list of indices covering the range of len(y)
        y: true responses
        y_pred: predicted responses
        y_label: name of response variable for given dataset
        dataset_name: name of dataset to add to plot title
        savepath: path to save our plot to

    Returns:
        None
    """

    # plot figure
    # what we are doing is plotting the index vs sales for both true and predicted responses
    # the extent to which they align together defines the strength of our predictions
    fig = plt.figure(figsize = (7, 7))
    ax = plt.axes()
    ax.set_facecolor("lavender")
    plt.plot(idx_range, y, c = 'b', linewidth = 2, linestyle = '-', label = 'true')
    plt.plot(idx_range, y_pred, c = 'r',  linewidth = 2, linestyle = '-', label = 'predictions')
    fig.suptitle(f'{dataset_name} - True vs. Predictions')    
    plt.xlabel('Index')                             
    plt.ylabel(f'{y_label}')
    plt.grid()                      
    plt.savefig(savepath)

    # return
    return None