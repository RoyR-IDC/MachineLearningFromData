
import numpy as np # used for scientific computing
import pandas as pd # used for data analysis and manipulation
import matplotlib.pyplot as plt # used for visualization and plotting

np.random.seed(42) 

# make matplotlib figures appear inline in the notebook
plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Read comma separated data
path = r"C:\MSC\ML\hw\HW1\data.csv"
df = pd.read_csv(path) # Make sure this cell runs regardless of your absolute path.
# df stands for dataframe, which is the default format for datasets in pandas

df.head(5)

X = df['sqft_living'].values
y = df['price'].values


"""
## Preprocessing

As the number of features grows, calculating gradients gets computationally expensive.
 We can speed this up by normalizing the input data to ensure all values are within the same range. 
 This is especially important for datasets with high standard deviations or differences in the ranges of the attributes. 
 Use [mean normalization](https://en.wikipedia.org/wiki/Feature_scaling) for the fearures (`X`) and the true labels (`y`).

Implement the cost function `preprocess`.
"""

def mean_normalization(data):
    data_shape  = data.shape
    data_normelized = data.copy()
    if data_shape.size == 1:
        data_shape.resize((data.size, 1))
        data_shape = data_shape.shape
    for i_feature in range (0, data_shape[1]):
        data_mean  = np.mean(data[:,i_feature])
        data_min  = np.min(data[:,i_feature])
        data_max  = np.max(data[:,i_feature])
        data_normelized[:,i_feature] = (data[:,i_feature]-data_mean)/(data_max-data_min)
    
    return data_normelized

def std_normalization(data):
    data_mean  = np.mean(data)
    data_std  = np.std(data)
    data_normelized = (data-data_mean)/(data_std)
    return data_normelized

def zero2one_normalization(data):
    data_min  = np.min(data)
    data_max  = np.max(data)
    data_normelized = (data-data_min)/(data_max-data_min)
    return data_normelized

def unit_normalization(data):
    data_sum  = np.sum(data)
    data_normelized = (data)/(data_sum)
    return data_normelized


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Inputs (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    
    X = mean_normalization(X)
    y = mean_normalization(y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y


X, y = preprocess(X, y)
###########################################################################
#                            START OF YOUR CODE                           #
###########################################################################
def add_bias_to_X(X):
    one_columns  = np.ones((X.shape[0],1))
    X = np.concatenate((one_columns, X), axis = 1)
    return X
X =  add_bias_to_X(X)
###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train], X[idx_val]
y_train, y_val = y[idx_train], y[idx_val]

plt.plot(X_train, y_train, 'ro', ms=1, mec='k') # the parameters control the size, shape and color of the scatter plot
plt.ylabel('Price in USD')
plt.xlabel('sq.ft')
plt.show()

"""
## Bias Trick

Make sure that 
`X` takes into consideration the bias $\theta_0$ in the linear model.
 Hint, recall that the predications of our linear model are of the form

$$
\hat{y} = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
$$

Add columns of ones as the zeroth column of the features 
(do this for both the training and validation sets).

"""





"""
## Part 2: Single Variable Linear Regression (40 Points)
Simple linear regression is a linear regression model with a single explanatory varaible and a single target value. 

$$
\hat{y} = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
$$

## Gradient Descent 

Our task is to find the best possible linear line that explains all the points in our dataset. We start by guessing initial values for the linear regression parameters $\theta$ and updating the values using gradient descent. 

The objective of linear regression is to minimize the cost function $J$:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{n}(h_\theta(x^{(i)})-y^{(i)})^2
$$

where the hypothesis (model) $h_\theta(x)$ is given by a **linear** model:

$$
h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
$$

$\theta_j$ are parameters of your model. and by changing those values accordingly you will be able to lower the cost function $J(\theta)$. One way to accopmlish this is to use gradient descent:

$$
\theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

In linear regresion, we know that with each step of gradient descent, the parameters $\theta_j$ get closer to the optimal values that will achieve the lowest cost $J(\theta)$.
"""


"""
Implement the cost function `compute_cost`. (10 points)

"""

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an obserbation's actual and
    predicted values for linear regression.  

    Input:
    - X: inputs  (n features over m instances).
    - y: true labels (1 value over m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # Use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    A = y 
    
    B = np.matmul(X, np.transpose(theta))
    mse = (np.square(A - B)).mean(axis=0)
    J = (mse/2)[0]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

theta = np.array([[-1, 2]])
J = compute_cost(X_train, y_train, theta)

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the *training set*. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    theta = theta.reshape((1,theta.size))
    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
    sample_size = y.size
    # this loop run on iterations
    for i_iteration in range(0, num_iters):
       error_vec = (np.matmul(theta,np.transpose(X)) - np.transpose(y)) # 1Xsample_size
       sumation = np.matmul(error_vec,X) # 1Xsample_size X sample_sizeX2 = 1X2
       new_theta_vec = theta - (alpha/sample_size)*sumation # 1X2 - 1X2
       current_J = compute_cost(X_train, y_train, new_theta_vec)
       J_history.append(current_J)
       theta = new_theta_vec.copy() # avoid changing the original thetas
       ########
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


np.random.seed(42)
theta = np.random.random(size=2)
iterations = 10
alpha = 0.1
theta, J_history = gradient_descent(X_train ,y_train, theta, alpha, iterations)

plt.figure()
plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.show()


def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the *training set*.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE np.linalg.pinv ##############
    """
    
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    X_T = np.transpose(X)
    X_T_X = np.matmul(X_T, X)
    inverse_X_T_X = np.linalg.inv(X_T_X)
    X_T_X_X_T = np.matmul(inverse_X_T_X, X_T)
    pinv_theta = np.transpose(np.matmul(X_T_X_X_T, y))
     
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

theta_pinv = pinv(X_train ,y_train)
J_pinv = compute_cost(X_train, y_train, theta_pinv)
"""
We can add the loss value for the theta calculated using the psuedo-inverse to our graph.
 This is another sanity check as the loss of our model should converge to the psuedo-inverse loss.
"""
plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
           linewidth = 1, linestyle = 'dashed')
plt.show()


"""
We can use a better approach for the implementation of `gradient_descent`.
 Instead of performing 40,000 iterations, we wish to stop when the improvement of the loss value
 is smaller than `1e-8` from one iteration to the next. Implement the function `efficient_gradient_descent`.
 (5 points)
"""


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the *training set*, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
  
    ###########################################################################
    desire_error = 1e-8
    theta = theta.reshape((1,theta.size))
    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
    sample_size = y.size
    # this loop run on iterations
    for i_iteration in range(0, num_iters):
       error_vec = (np.matmul(theta,np.transpose(X)) - np.transpose(y)) # 1Xsample_size
       sumation = np.matmul(error_vec,X) # 1Xsample_size X sample_sizeX2 = 1X2
       new_theta_vec = theta - (alpha/sample_size)*sumation # 1X2 - 1X2
       current_J = compute_cost(X_train, y_train, new_theta_vec)
       J_history.append(current_J)
       theta = new_theta_vec.copy() # avoid changing the original thetas
       if desire_error > current_J:
           break
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

"""
The learning rate is another factor that determines the performance of our model in terms of speed and accuracy. 
Complete the function `find_best_alpha`. Make sure you use the training dataset to learn the parameters 
(thetas) and use those parameters with the validation dataset to compute the cost.

"""


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over provided values of alpha and train a model using the 
    *training* dataset. maintain a python dictionary with alpha as the 
    key and the loss on the *validation* set as the value.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {key (alpha) : value (validation loss)}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    ###########################################################################
   

    for current_alpha in alphas:
        initial_theta = np.array([[-1, 2]])
        suggested_theta, J_history = efficient_gradient_descent(X_train, y_train, initial_theta, current_alpha, iterations)
        validation_J_for_specific_alphas = compute_cost(X_val, y_val, suggested_theta)
        alpha_dict[current_alpha] =  validation_J_for_specific_alphas
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

alpha_dict = find_best_alpha(X_train, y_train, X_val, y_val, 10)

"""
Obtain the best learning rate from the dictionary `alpha_dict`. This can be done in a single line using built-in functions.
"""

###########################################################################
#                            START OF YOUR CODE                           #
###########################################################################
result = alpha_dict.items()
  
# Convert object to a list
data = list(result)
  
# Convert list to an array
data_array = np.array(data)
mse_values = data_array[:,1]
alpha_values = data_array[:,0]

best_alpha_index = np.where(np.min(mse_values) == mse_values)[0]
best_alpha  = alpha_values[best_alpha_index]
best_mse  = mse_values[best_alpha_index]

###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################
print(best_alpha)

"""
Pick the best three alpha values you just calculated and provide **one** 
graph with three lines indicating the training loss as a function of iterations (Use 10,000 iterations). 
Note you are required to provide general code for this purpose (no hard-coding). Make sure the visualization is 
clear and informative. (5 points)
"""

###########################################################################
#                            START OF YOUR CODE                           #
###########################################################################
amount_of_numbers = 3
three_best_mse_index = mse_values.argsort()[0:3][::-1]
best_alphas = np.take(alpha_values, three_best_mse_index)
desire_amount_of_iterations = 100
alphas = best_alphas
alpha_dict = {}
###########################################################################


for current_alpha in alphas:
    initial_theta = np.array([[-1, 2]])
    suggested_theta, J_history = efficient_gradient_descent(X_train, y_train, initial_theta, current_alpha, iterations)
    validation_J_for_specific_alphas = compute_cost(X_val, y_val, suggested_theta)
    alpha_dict[current_alpha] =  validation_J_for_specific_alphas
    
    
###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################


"""
This is yet another sanity check. This function plots the regression 
lines of your model and the model based on the pseudoinverse calculation. 
Both models should exhibit the same trend through the data. 
"""



plt.figure(figsize=(7, 7))
plt.plot(X_train[:,1], y_train, 'ro', ms=1, mec='k')
plt.ylabel('Price in USD')
plt.xlabel('sq.ft')
plt.plot(X_train[:, 1], np.dot(X_train, theta), 'o')
plt.plot(X_train[:, 1], np.dot(X_train, theta_pinv), '-')

plt.legend(['Training data', 'Linear regression', 'Best theta']);



"""
## Part 2: Multivariate Linear Regression (30 points)

In most cases, you will deal with databases that have more than one feature. 
It can be as little as two features and up to thousands of features. In those cases,
we use a multiple linear regression model. 
The regression equation is almost the same as the simple linear regression equation:

$$
\hat{y} = h_\theta(\vec{x}) = \theta^T \vec{x} = \theta_0 + \theta_1 x_1 + ... + \theta_n x_n
$$


If you wrote vectorized code, this part should be straightforward. If your code is not vectorized, you should go back and edit your functions such that they support both multivariate and single variable regression. **Your code should not check the dimensionality of the input before running**.

"""

# Read comma separated data
df = pd.read_csv(path)
df.head()

"""
## Preprocessing

Like in the single variable case, we need to create a numpy array from the dataframe.
 Before doing so, we should notice that some of the features are clearly irrelevant.


"""


X = df.drop(columns=['price', 'id', 'date']).values
y = df['price'].values


# preprocessing
X, y = preprocess(X, y)


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train,:], X[idx_val,:]
y_train, y_val = y[idx_train], y[idx_val]


import mpl_toolkits.mplot3d.axes3d as p3
fig = plt.figure(figsize=(5,5))
ax = p3.Axes3D(fig)
xx = X_train[:, 1][:1000]
yy = X_train[:, 2][:1000]
zz = y_train[:1000]
ax.scatter(xx, yy, zz, marker='o')
ax.set_xlabel('bathrooms')
ax.set_ylabel('sqft_living')
ax.set_zlabel('price')
plt.show()

"""
Use the bias trick again (add a column of ones as the zeroth column in the both the training and validation datasets).
"""

###########################################################################
#                            START OF YOUR CODE                           #
###########################################################################
X =  add_bias_to_X(X)
###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################


"""
Make sure the functions `compute_cost` (10 points),
`gradient_descent` (15 points), and `pinv` (5 points) work on the multi-dimensional dataset.
lk If you make any changes, make sure your code still works on the single variable regression model. 
"""
np.random.seed(42)
shape = X_train.shape[1]
theta = np.random.random(shape)
iterations = 40000
theta, J_history = gradient_descent(X_train ,y_train, theta, best_alpha, iterations)

theta_pinv = pinv(X_train ,y_train)
J_pinv = compute_cost(X_train, y_train, theta_pinv)

"""
We can use visualization to make sure the code works well. 
Notice we use logarithmic scale for the number of iterations, 
since gradient descent converges after ~500 iterations.

"""


plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations - multivariate linear regression')
plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
           linewidth = 1, linestyle = 'dashed')
plt.show()


"""
## Part 3: Polynomial Regression (10 points)

Linear Regression allows us to explore linear relationships but if we need a model
 that describes non-linear dependencies we can also use Polynomial Regression.
 In order to perform polynomial regression, we create additional features using a function of the original 
 features and use standard linear regression on the new features. For example, consider the following single
 variable $(x)$ cubic regression:

$$ x_0 = 1, \space x_1 = x, \space x_2 = x^2, \space x_3 = x^3$$

And after using standard linear regression:

$$ f(x) = \theta_0 + \theta_1 x + \theta_2 x^2 +  \theta_3 x^3$$

As required. 

For this exercise, use polynomial regression by using all **quadratic** feature combinations: 

$$ 1, x, y, z, x^2, y^2, z^2, xy, xz, yz, ...$$

and evaluate the MSE cost on the training and testing datasets.

"""


columns_to_drop = ['price', 'id', 'date']
all_features = df.drop(columns=columns_to_drop)
all_features.head(5)
from sympy import expand, symbols


amount_of_features = all_features.shape[1]
variable_dict = {}
vars_string = ' '.join(['%c' % x for x in range(97, 97+amount_of_features)])  # gives 'abcdefghij'
variable = symbols(vars_string)
for i_var_idx in range(amount_of_features):
    if i_var_idx == 0 :
        gfg_exp = variable[i_var_idx]
    else:
        
        gfg_exp += variable[i_var_idx]

exp = sympy.expand(gfg_exp**2)



 






















"""
Give an explanations to the results and compare them to regular linear regression. Do they make sense?
"""

"""
### Use this Markdown cell for your answer
"""

"""
## Part 4: Adaptive Learning Rate (10 points)

So far, we kept the learning rate alpha constant during training. However, changing alpha during training might improve convergence in terms of the global minimum found and running time. Implement the adaptive learning rate method based on the gradient descent algorithm above. 

**Your task is to find proper hyper-parameter values for the adaptive technique and compare this technique to the constant learning rate. Use clear visualizations of the validation loss and the learning rate as a function of the iteration**. 

Time based decay: this method reduces the learning rate every iteration according to the following formula:

$$\alpha = \frac{\alpha_0}{1 + D \cdot t}$$

Where $\alpha_0$ is the original learning rate, $D$ is a decay factor and $t$ is the current iteration.
"""

### Your code here ###






