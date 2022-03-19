import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make matplotlib figures appear inline in the notebook
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, node):
        self.children.append(node)
        
        
n = Node(5)
p = Node(6)
q = Node(7)
n.add_child(p)
n.add_child(q)
n.children


# load dataset
path  = r'C:\MSC\ML\hw\HW2\agaricus-lepiota.csv'
data = pd.read_csv('agaricus-lepiota.csv')


#############################################################################
# TODO: Find columns with missing values and remove them from the data.#
#############################################################################
pass
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################


"""
We will split the dataset to `Training` and `Testing` datasets.
"""
from sklearn.model_selection import train_test_split
# Making sure the last column will hold the labels
X, y = data.drop('class', axis=1), data['class']
X = np.column_stack([X,y])
# split dataset using random_state to get the same split each time
X_train, X_test = train_test_split(X, random_state=99)

print("Training dataset shape: ", X_train.shape)
print("Testing dataset shape: ", X_test.shape)

print(y.shape)
def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns the gini impurity.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

##### Your Tests Here #####
calc_gini(X), calc_entropy(X)


"""
## Goodness of Split

Given a feature the Goodnees of Split measures the reduction in the impurity 
if we split the data according to the feature.
$$
\Delta\varphi(S, A) = \varphi(S) - \sum_{v\in Values(A)} \frac{|S_v|}{|S|}\varphi(S_v)
$$

NOTE: you can add more parameters to the function and you can also add more 
returning variables (The given parameters and the given returning variable should not be touch). (10 Points)
"""
def goodness_of_split(data, feature, impurity_func):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.

    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index.
    - impurity func: a function that calculates the impurity.

    Returns the goodness of split.  
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    goodness = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness    



"""
## Building a Decision Tree

Use a Python class to construct the decision tree. Your class should 
support the following functionality:

1. Initiating a node for a decision tree. You will need to use several 
class methods and class attributes and you are free to use them as you see fit. 
We recommend that every node will hold the 
feature and value used for the split and its children.
2. Your code should support both Gini and Entropy as impurity measures. 
3. The provided data includes categorical data. In this exercise, 
when splitting a node create the number of children needed
 according to the attribute unique values.

Complete the class `DecisionNode`. The structure of this class is entirely up to you. 

Complete the function `build_tree`. 
This function should get the training dataset and the impurity as inputs, 
initiate a root for the decision tree 
and construct the tree according to the procedure you learned in class. (30 points)

"""
class DecisionNode:
    """
    This class will hold everything you require to construct a decision tree.
    The structure of this class is up to you. However, you need to support basic 
    functionality as described above. It is highly recommended that you 
    first read and understand the entire exercise before diving into this class.
    """
    def __init__(self, feature):
        self.feature = feature # column index of criteria being tested
        
    def add_child(self, node):
        self.children.append(node)
        
        
        
def build_tree(data, impurity, min_samples_split=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - min_samples_split: the minimum number of samples required to split an internal node
    - max_depth: the allowable depth of the tree

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root
        

# python supports passing a function as an argument to another function.
tree_gini = build_tree(data=X_train, impurity=calc_gini) # gini and goodness of split
tree_entropy = build_tree(data=X_train, impurity=calc_entropy) # entropy and goodness of split
tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True) # entropy and gain ratio
        
        

"""
## Tree evaluation

Complete the functions `predict` and `calc_accuracy`. (10 points)
"""

def predict(node, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return node.pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy 


"""
After building the three trees using the training set, 
you should calculate the accuracy on the test set. For each tree print the
training and test accuracy. Select the tree that gave you the best test accuracy. 
For the rest of the exercise, use that tree (when you asked to build another 
tree use the same impurity function and same gain_ratio flag). 
"""
    


#### Your code here ####



#### Your code here ####


"""
## Depth pruning

(15 points)

Consider the following max_depth values: [1, 2, 3, 4, 5, 6, 7, 8]. 
For each value, construct a tree and prune it 
according to the max_depth value = don't let the tree to grow
 beyond this depth. Next, calculate the training and testing accuracy.<br>
On a single plot, draw the training and testing accuracy as a function 
of the max_depth. Mark the best result on the graph with red circle.
"""

#### Your code here ####


#### Your code here ####


"""
## Min Samples Split

(15 points)

Consider the following min_samples_split values: [1, 5, 10, 20, 50].
 For each value, construct a tree and prune it according 
 to the min_samples_split value = don't split a node if the number of
 sample in it is less or equal to the min_samples_split value. Next, 
 calculate the training and testing accuracy.<br>
On a single plot, draw the training and testing accuracy as a
 function of the min_samples_split. Mark the best result on the
 graph with red circle. (make sure that the x-axis ticks represent
 the values of min_samples_split)
"""

#### Your code here ####


#### Your code here ####


"""
Build the best 2 trees:
1. tree_max_depth - the best tree according to max_depth pruning
1. tree_min_samples_split - the best tree according to min_samples_split pruning
"""

#### Your code here ####

#### Your code here ####


"""
## Number of Nodes

(5 points)

Complete the function counts_nodes and print the number of
nodes in each tree and print the number of nodes of the two trees above
"""

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    
    
    
"""
## Print the tree

Complete the function `print_tree`. Your tree should be visualized clearly. You can use the following example as a reference:
```
[ROOT, feature=X0],
  [X0=a, feature=X2]
    [X2=c, leaf]: [{1.0: 10}]
    [X2=d, leaf]: [{0.0: 10}]
  [X0=y, feature=X5], 
    [X5=a, leaf]: [{1.0: 5}]
    [X5=s, leaf]: [{0.0: 10}]
  [X0=e, leaf]: [{0.0: 25, 1.0: 50}]
```
In each brackets:
* The first argument is the parent feature with the value that led to current node
* The second argument is the selected feature of the current node
* If the current node is a leaf, you need to print also the labels and their counts

(5 points)
"""

# you can change the function signeture
def print_tree(node, depth=0, parent_feature='ROOT', feature_val='ROOT'):
    '''
    prints the tree according to the example above

    Input:
    - node: a node in the decision tree

    This function has no return value
    '''
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    
"""
print the tree with the best test accuracy and with less 
than 50 nodes (from the two pruning methods)
"""



#### Your code here ####



#### Your code here ####




