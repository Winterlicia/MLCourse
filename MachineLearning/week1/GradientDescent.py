import math, copy
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
from CostFunction import computeCostFunction

# Same two data points
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# For gradient descent to work, update w,b simultaneously until convergence

# COMPUTE GRADIENT 
    # computes the derivative terms for w and b

def compute_gradient(x_train, y_train, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x_train (ndarray (m,)): Data, m examples 
      y_train (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x_train.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    # Compute the gradient as per the formula:
    '''
    dJ/dw = 1/m * summation of i to m (Fwb(x) - y) * x
    dJ/db = 1/m * summation of i to m (Fwb(x) - y) 
    '''
    for i in range(m):  
        f_wb = w * x_train[i] + b 
        dj_dw_i = (f_wb - y_train[i]) * x_train[i] 
        dj_db_i = f_wb - y_train[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 

    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db