import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make matplotlib figures appear inline in the notebook
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

        
        
        
        
# n = Node(5)
# p = Node(6)
# q = Node(7)
# n.add_child(p)
# n.add_child(q)
# n.children
global columns_list, header
# load dataset
path = r'C:\MSC\ML\hw\HW2\agaricus-lepiota.csv'
data = pd.read_csv(path)
columns_list =  data.columns.to_list()

# normerization
data = data.applymap(lambda x: ord(x))
data.columns = columns_list


data = data[0:100]

#####################################################3

data = data.to_numpy().tolist()
header = columns_list

# Column labels.
# These are used only to print the tree.
def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        string2return  = "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))
        # feature_name = header[self.column]
        #return string2return
        return string2return

def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows
def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(data)
    gini = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(data))
        gini -= prob_of_lbl**2
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
    counts = class_counts(data)
    entropy = 0.0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(data))
        entropy -= prob_of_lbl*np.log2(prob_of_lbl)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * calc_gini(left) - (1 - p) * calc_gini(right)

def find_best_split(rows, impurity):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information impurity_gain."""
    best_gain = 0  # keep track of the best information impurity_gain
    best_question = None  # keep train of the feature / value that produced it
    best_feature_name = ''
    current_uncertainty = impurity(rows)


    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value
            question = Question(col, val)
            feature_name = str(question).split(' ')[1]
           

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information impurity_gain from this split
            impurity_gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if impurity_gain >= best_gain:
                best_gain, best_question, best_feature_name = impurity_gain, question, feature_name

    return best_gain, best_question, best_feature_name


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []
        self.predictions = class_counts(data)

    def add_child(self, node):
        self.children.append(node)

class DecisionNode:
    """
    This class will hold everything you require to construct a decision tree.
    The structure of this class is up to you. However, you need to support basic 
    functionality as described above. It is highly recommended that you 
    first read and understand the entire exercise before diving into this class.
    """
    def __init__(self, feature):
        self.feature = feature # column index of criteria being tested
        self.children = [] # create children list

    def add_child(self, node):
        self.children.append(node)
        
    def update_question_and_children(self, question, true_branch,false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        


def build_tree(data, impurity, gain_ratio=False, min_samples_split=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or impurity_gain ratio flag
    - min_samples_split: the minimum number of samples required to split an internal node
    - max_depth: the allowable depth of the tree

    Output: the root node of the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information impurity_gain,
    # and return the question that produces the highest impurity_gain.
    impurity_gain, question , feature_name  = find_best_split(data, impurity)

    # Base case: no further info impurity_gain
    # Since we can ask no further questions,
    # we'll return a Node.
    if impurity_gain == 0:
        return Node(data)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(data, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, impurity, gain_ratio, min_samples_split, max_depth)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, impurity, gain_ratio, min_samples_split, max_depth)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    decision_node  = DecisionNode(feature_name)
    decision_node.update_question_and_children(question, true_branch,false_branch)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return decision_node

gini_tree = build_tree(data, calc_entropy, gain_ratio=False)
a=5

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a Node
    if isinstance(node, Node):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

print_tree(gini_tree)

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a Node
    if isinstance(node, Node):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)
#######
# # Demo:
# # The tree predicts the 1st row of our
# # training data is an apple with confidence 1.
# classify(training_data[0], my_tree)
# #######
def print_Node(counts):
    """A nicer way to print the predictions at a Node."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs
# #######
# # Demo:
# # Printing that a bit nicer
# print_Node(classify(training_data[0], my_tree))
# #######
# #######
# # Demo:
# # On the second example, the confidence is lower
# print_Node(classify(training_data[1], my_tree))
# #######
# # Evaluate
# testing_data = [
#     ['Green', 3, 'Apple'],
#     ['Yellow', 4, 'Apple'],
#     ['Red', 2, 'Grape'],
#     ['Red', 1, 'Grape'],
#     ['Yellow', 3, 'Lemon'],
# ]
# for row in testing_data:
#     print ("Actual: %s. Predicted: %s" %
#            (row[-1], print_Node(classify(row, my_tree))))
##########################################################



#############################################################################
# TODO: Find columns with missing values and remove them from the data.#
#############################################################################
columns_with_empty_value_list = data.columns[data.isna().any()].tolist()
if columns_with_empty_value_list.__len__():
    remove_columns_string = ', '.join(columns_with_empty_value_list)
    print('The following columns [' + remove_columns_string + \
          '] have missing values in data, and therefore removed')
    data = data.drop(columns_with_empty_value_list)
else:
    print('There is no missing values, and therefore no columns has been removed')
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################


"""
We will split the dataset to `Training` and `Testing` datasets.
"""
from sklearn.model_selection import train_test_split

# Making sure the last column will hold the labels
X, y = data.drop('class', axis=1), data['class']
X = np.column_stack([X, y])

# split dataset using random_state to get the same split each time
X_train, X_test = train_test_split(X, random_state=99)

print("Training dataset shape: ", X_train.shape)
print("Testing dataset shape: ", X_test.shape)

print(y.shape)





##### Your Tests Here #####
calc_gini(X)
calc_entropy(X)


def gen_info_gain(data, feature, impurity_func):
    impurity_before_changing = impurity_func(data)
    data_feature = data[feature]
    amount_of_values = data_feature.shape[0]

    #
    separation_dict, separation_dict_L_S = find_split_threshould(data_feature, impurity_func)

    dict_is_empty = not bool(separation_dict)
    split_flag = True
    if dict_is_empty:
        # mpurity does not change
        info_gain = impurity_before_changing
        th_value =  0
        split_flag = False
    else:
        dict_values = separation_dict.items()
        dict_values = list(dict_values)
        dict_values = np.array(dict_values)
        best_th_index = np.where(dict_values[:, 1] == np.max(dict_values[:, 1]))[0][0]
        th_value = dict_values[best_th_index, 0]
        large = separation_dict_L_S[str(th_value) + '_L']
        small = separation_dict_L_S[str(th_value) + '_S']
        info_gain = (impurity_before_changing - large - small)
        if large == 0 and small == 0 :
            split_flag = False

        # print(large)
        # print(small)

    return info_gain, separation_dict, separation_dict_L_S, th_value, split_flag


def find_split_threshould(data_feature, impurity_func):
    data_feature_count = data_feature.value_counts()
    data_feature_values = data_feature_count.index.to_list()
    data_feature_values.sort()
    separation_dict = {}
    separation_dict_L_S = {}
    for i_value in range(0, data_feature_values.__len__() - 1):
        split_values = data_feature_values[i_value:i_value + 2]
        split_value = np.mean(split_values)
        if split_value in split_values:
            continue
        else:
            data_feature_values_array = np.array(data_feature_values)
            larger_index = (data_feature_values_array >= split_value)
            larger_array = data_feature_values_array[larger_index]
            smaller_array = data_feature_values_array[~larger_index]

            amount_of_features = data_feature_values_array.size
            amount_smaller = smaller_array.size
            amount_larger = larger_array.size

            larger_value = (amount_larger / amount_of_features) * impurity_func(larger_array)
            smaller_value = (amount_smaller / amount_of_features) * impurity_func(smaller_array)
            goodness_of_threshold = (1 - larger_value - smaller_value)
            separation_dict[split_value] = goodness_of_threshold
            separation_dict_L_S[str(split_value) + '_L'] = larger_value
            separation_dict_L_S[str(split_value) + '_S'] = smaller_value
            try:
                separation_dict_L_S[str(split_value) + 'L_split_ratio'] = (amount_larger / amount_of_features)
                separation_dict_L_S[str(split_value) + 'S_split_ratio'] = (amount_smaller / amount_of_features)
            except:
                a=5
    return separation_dict, separation_dict_L_S
def generate_split_information_rate(separation_dict_L_S, th_value):
    L_split_ratio = separation_dict_L_S[str(th_value) + 'L_split_ratio']
    S_split_ratio = separation_dict_L_S[str(th_value) + 'S_split_ratio']
    split_information = -(np.log2(L_split_ratio) + np.log2(S_split_ratio))
    return split_information



"""
from scipy.stats import entropy
entropy([1/2, 1/2], base=2)


"""
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


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.

    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index.
    - impurity func: a function that calculates the impurity.
    - gain_ratio: goodness of split or impurity_gain ratio flag.

    Returns the goodness of split (or the Gain Ration).
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    ###########################################################################
    goodness = 0
    info_gain, separation_dict, separation_dict_L_S, th_value, split_flag = gen_info_gain(data, feature, calc_entropy)
    if gain_ratio:  # Gain ratio
        split_information = generate_split_information_rate(separation_dict_L_S, th_value)
        goodness = (info_gain / split_information)
    else:  # Goodness of split
        goodness = info_gain
    #                   END OF YOUR CODE                            #
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



# python supports passing a function as an argument to another function.
tree_gini = build_tree(data=X_train, impurity=calc_gini)  # gini and goodness of split
tree_entropy = build_tree(data=X_train, impurity=calc_entropy)  # entropy and goodness of split
tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True)  # entropy and impurity_gain ratio

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
    [X2=c, Node]: [{1.0: 10}]
    [X2=d, Node]: [{0.0: 10}]
  [X0=y, feature=X5], 
    [X5=a, Node]: [{1.0: 5}]
    [X5=s, Node]: [{0.0: 10}]
  [X0=e, Node]: [{0.0: 25, 1.0: 50}]
```
In each brackets:
* The first argument is the parent feature with the value that led to current node
* The second argument is the selected feature of the current node
* If the current node is a Node, you need to print also the labels and their counts

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
