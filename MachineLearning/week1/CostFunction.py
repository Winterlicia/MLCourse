# IMPLEMENT AND EXPLORE COST FUNCTION FOR LINEAR REGRESSION FOR ONE VARIABLE

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

'''
EQUATION:

J(w,b) = 1/2m * summation from 0 to m of (modelFunction - y^i)^2
Summations compute error
'''

def computeCostFunction(x_train, y_train, slope, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x_train (ndarray (m,)): Data, m examples 
      y_train (ndarray (m,)): target values
      slope, b (scalar)    : model parameters  
    
    Returns
        totalCost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x_train.shape[0]
    summation = 0
    for i in range(m):
        modelFwb = slope * x_train[i] + b
        summation += (modelFwb - y_train[i])**2
    
    return summation / (2*m)

# COST FUNCTION INTUITION
    # Cost measures how accurate the model is on the training data
    # Cost eqn shows that if w & b can be selected such that Fwb = y^i, the summation terms will be zero and the cost is minimied 
    # b = 100 provides optimal solution, so focus on w
    # If w is set to 200, cost is minimized which matches results in ModelRepresentation

# plt.intuition(x_train, y_train)

# COST FUNCTION VISUALIZATION - 3D
    # See how Cost function varies with respect to both w and b in 3D or Contour plot
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
plt.show()

'''
 Dashed lines in the left plot represent the portion of the cost contributed by each example in your training set. 
 In this case, values of approximately w = 209, b = 2.4 provide low cost. 
 Note that, because our training examples are not on a line, the minimum cost is not zero.
'''

# CONVEX COST SURFACE
    # Cost function square the loss ensures that 'error surface' is convex like a soup bowl. 
    # Minimum can be reached by following the gradient in all dimensions
    # B/c w and b dimensions scale differently, this is not easy to recognize

soup_bowl()
