# ml-scratch-linear-regression
Linear Regression Algorithm

## **Description**
The following is my from scratch implementation of the Linear Regression algorithm.

### **Dataset**

For datasets I used three datasets: \
\
    &emsp;1. Simple X vs. y Dataset \
    &emsp;2. TV Marketing Dataset \
    &emsp;3. BMI and Life Expectancy Dataset \
\
For each dataset I load it and scale the predictor and respone variables to the range [0, 1]. This is to avoid the magnitude differences that can arise during the fitting process.

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, and matplotlib.

**2.** Once you made sure all these libraries are installed, evrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the three datasets \
    &emsp;**ii.** Split data into train and test sets \
    &emsp;**iii.** Build a linear regressor \
    &emsp;**iv.** Fit the linear regressor \
    &emsp;**v.** Predict on the test set \
    &emsp;**vi.** Plot the scatter plot, fitted line, loss curve, and prediction alignments.

**4.** In main.py I specify a set of hyperparameters, these can be picked by the user. The main ones worth noting are the number of epochs and learning rate. These hyperparameters were chosen through trail & error experimentation on each dataset.

### **Results**

For each dataset I will list the number of epochs, learning rate, and test MSE loss score.
In addition I offer four visualization plots for a better assessment.

**1.** Simple X vs. y Dataset:

- Hyperparameters:
     - Number of epochs = 1000
     - Learning rate = 0.1
 
- Numerical Result:
     - MSE Test Loss = 0.007

- See visualizations below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/xy/xy_scatter.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/xy/xy_loss.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/xy/xy_fitted.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/xy/xy_alignment.png?raw=true)

**2.** TV Marketing Dataset:

- Hyperparameters:
     - Number of epochs = 100
     - Learning rate = 0.4
 
- Numerical Result:
     - MSE Test Loss = 0.0139

- See visualizations below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/tv/tv_scatter.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/tv/tv_loss.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/tv/tv_fitted.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/tv/tv_alignment.png?raw=true)

**3.** BMI and Life Expectancy Dataset:

- Hyperparameters:
     - Number of epochs = 1000
     - Learning rate = 0.1
 
- Numerical Result:
     - MSE Test Loss = 0.0299

- See visualizations below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/bmi/bmi_scatter.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/bmi/bmi_loss.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/bmi/bmi_fitted.png?raw=true)

![alt text](https://github.com/ZainUFarhat/ml-scratch-linear-regression/blob/main/plots/bmi/bmi_alignment.png?raw=true)