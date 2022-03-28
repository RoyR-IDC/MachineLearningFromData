"""
# Exercise 2: Decision Trees

In this assignment you will implement a Decision Tree algorithm as learned in class.

## Read the following instructions carefully:

1. This jupyter notebook contains all the step by step instructions needed for this exercise.
1. Submission includes this notebook only with the exercise number and your ID as the filename. For example: `hw2_123456789_987654321.ipynb` if you submitted in pairs and `hw2_123456789.ipynb` if you submitted the exercise alone.
1. Write **efficient vectorized** code whenever possible. Some calculations in this exercise take several minutes when implemented efficiently, and might take much longer otherwise. Unnecessary loops will result in point deduction.
1. You are responsible for the correctness of your code and should add as many tests as you see fit. Tests will not be graded nor checked.
1. Write your functions in this notebook only. **Do not create Python modules and import them**.
1. You are allowed to use functions and methods from the [Python Standard Library](https://docs.python.org/3/library/) and [numpy](https://www.numpy.org/devdocs/reference/) only. **Do not import anything else.**
1. Your code must run without errors. Make sure your `numpy` version is at least 1.15.4 and that you are using at least python 3.6. Changes of the configuration we provided are at your own risk. Any code that cannot run will not be graded.
1. Write your own code. Cheating will not be tolerated.
1. Answers to qualitative questions should be written in **markdown** cells (with $\LaTeX$ support). Answers that will be written in commented code blocks will not be checked.

## In this exercise you will perform the following:
1. Practice OOP in python.
2. Implement two impurity measures: Gini and Entropy.
3. Construct a decision tree algorithm.
4. Prune the tree to achieve better results.
5. Visualize your results.
"""
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



def unique_vals(datas, col):
    """Find the unique values for a column in a dataset."""
    return set([data[col] for data in datas])

def class_counts(datas):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for data in datas:
        # in our dataset format, the label is always the last column
        label = data[-1]
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

def partition(datas, question):
    """Partitions a dataset.

    For each data in the dataset, check if it matches the question. If
    so, add it to 'true datas', otherwise, add it to 'false datas'.
    """
    true_datas, false_datas = [], []
    for data in datas:
        if question.match(data):
            true_datas.append(data)
        else:
            false_datas.append(data)
    return true_datas, false_datas

def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    info_gain = (current_uncertainty - p * calc_gini(left) - (1 - p) * calc_gini(right))
    split_information_rate = -np.log2(p)*np.log2(1-p)
    return info_gain, split_information_rate

def find_best_split(datas, impurity):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information impurity_gain."""
    best_gain = 0  # keep track of the best information impurity_gain
    best_question = None  # keep train of the feature / value that produced it
    best_feature_name = ''
    best_split_information_rate = 0
    current_uncertainty = impurity(datas)


    n_features = len(datas[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([data[col] for data in datas])  # unique values in the column

        for val in values:  # for each value
            question = Question(col, val)
            feature_name = str(question).split(' ')[1]
           

            # try splitting the dataset
            true_datas, false_datas = partition(datas, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_datas) == 0 or len(false_datas) == 0:
                continue

            # Calculate the information impurity_gain from this split
            impurity_gain, split_information_rate = info_gain(true_datas, false_datas, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if impurity_gain >= best_gain:
                best_gain, best_question, best_feature_name, best_split_information_rate = impurity_gain, question, feature_name, split_information_rate

    return best_gain, best_question, best_feature_name, best_split_information_rate



class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []
        if not np.isscalar(data):
            self.predictions = class_counts(data)
        else:
            self.predictions = {data:1}

    def add_child(self, node):
        self.children.append(node)
        
n = Node(5)
p = Node(6)
q = Node(7)
n.add_child(p)
n.add_child(q)
n.children


"""
## Data preprocessing

For the following exercise, we will use a dataset containing mushroom data `agaricus-lepiota.csv`. 

This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous
one (=there are only two classes **edible** and **poisonous**). 
    
The dataset contains 8124 observations with 22 features:
1. cap-shape: bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
4. bruises: bruises=t,no=f
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
6. gill-attachment: attached=a,descending=d,free=f,notched=n
7. gill-spacing: close=c,crowded=w,distant=d
8. gill-size: broad=b,narrow=n
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
10. stalk-shape: enlarging=e,tapering=t
11. stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
16. veil-type: partial=p,universal=u
17. veil-color: brown=n,orange=o,white=w,yellow=y
18. ring-number: none=n,one=o,two=t
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
21. population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
22. habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

First, we will read and explore the data using pandas and the `.read_csv` method. Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
"""
        
        
global columns_list, header
# load dataset
path = r'C:\MSC\ML\hw\HW2\agaricus-lepiota.csv'
data = pd.read_csv(path)
data = data[0:200] # to remove 
columns_list =  data.columns.to_list()


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


# normerization
do_nomerzation = False
if do_nomerzation:
    data = data.applymap(lambda x: ord(x))
data.columns = columns_list
header = columns_list
#data = data.to_numpy().tolist()



from sklearn.model_selection import train_test_split
# Making sure the last column will hold the labels
X, y = data.drop('class', axis=1), data['class']
X = np.column_stack([X,y])
# split dataset using random_state to get the same split each time
X_train, X_test = train_test_split(X, random_state=99)

print("Training dataset shape: ", X_train.shape)
print("Testing dataset shape: ", X_test.shape)


y.shape


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
    """Calculate the Gini Impurity for a list of datas.

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



##### Your Tests Here #####
calc_gini(X), calc_entropy(X)

"""

Given a feature the Goodnees of Split measures the reduction in the impurity if we split the data according to the feature.
$$
\Delta\varphi(S, A) = \varphi(S) - \sum_{v\in Values(A)} \frac{|S_v|}{|S|}\varphi(S_v)
$$

In our implementation the goodness_of_split function will return either the Goodness of Split or the Gain Ratio as learned in class. You'll control the return value with the `gain_ratio` parameter. If this parameter will set to False (the default value) it will return the regular Goodness of Split. If it will set to True it will return the Gain Ratio.
$$
GainRatio(S,A)=\frac{InformationGain(S,A)}{SplitInformation(S,A)}
$$
Where:
$$
InformationGain(S,A)=Goodness\ of\ Split\ calculated\ with\ Entropy\ as\ the\ Impurity\ function \\
SplitInformation(S,A)=- \sum_{a\in A} \frac{|S_a|}{|S|}\log\frac{|S_a|}{|S|}
$$
NOTE: you can add more parameters to the function and you can also add more returning variables (The given parameters and the given returning variable should not be touch). (10 Points)
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
    impurity_gain, question , feature_name, split_information_rate  = find_best_split(data, impurity_func)

    if gain_ratio:  # Gain ratio
        goodness = (impurity_gain / split_information_rate)
    else:  # Goodness of split
        goodness = impurity_gain
    # END OF YOUR CODE                                                        #
    ###########################################################################
    return goodness


"""
## Building a Decision Tree

Use a Python class to construct the decision tree. Your class should support the following functionality:

1. Initiating a node for a decision tree. You will need to use several class methods and class attributes and you are free to use them as you see fit. We recommend that every node will hold the feature and value used for the split and its children.
2. Your code should support both Gini and Entropy as impurity measures. 
3. The provided data includes categorical data. In this exercise, when splitting a node create the number of children needed according to the attribute unique values.

Complete the class `DecisionNode`. The structure of this class is entirely up to you. 

Complete the function `build_tree`. This function should get the training dataset and the impurity as inputs, initiate a root for the decision tree and construct the tree according to the procedure you learned in class. (30 points)
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
    You are required to fully gdata the tree until all leaves are pure. 

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
    impurity_gain, question , feature_name, split_information_rate  = find_best_split(data, impurity)

    # Base case: no further info impurity_gain
    # Since we can ask no further questions,
    # we'll return a Node.
    if impurity_gain == 0:
        return Node(data)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_datas, false_datas = partition(data, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_datas, impurity, gain_ratio, min_samples_split, max_depth)

    # Recursively build the false branch.
    false_branch = build_tree(false_datas, impurity, gain_ratio, min_samples_split, max_depth)

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
    # Base case: we've reached a leaf
    if isinstance(node, Node):
        dict_values = node.predictions.items()
        dict_values = list(dict_values)
        dict_values = np.array(dict_values)
        if is_numeric(dict_values[:,1]):
            max_indexs = np.where(dict_values[:,1] == np.max(dict_values[:,1]))[0]
        else:
            prediection_values = np.float32(dict_values[:,1])
            max_indexs = np.where(prediection_values == np.max(prediection_values))[0]

        if max_indexs.size >1:
            max_indexs = np.random.choice(max_indexs, 1)
        prediction = dict_values[max_indexs[0],0]
        return prediction

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(instance):
        return predict(node.true_branch, instance)
    else:
        return predict(node.false_branch, instance)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

predict(tree_entropy, X_train[0,:].tolist())  # example 

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
    error_sum = 0
    for i_row in dataset:
        y_pred = predict(node, i_row)  
        y_true = i_row[-1]
        if y_true != y_pred:
            add_error = 1
        else:
            add_error = 0
        error_sum += add_error
    error_loss = (1/dataset.shape[1])*error_sum
    accuracy = (1-error_loss)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    return accuracy 


calc_accuracy(tree_entropy, X_test)




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
    """World's most elegant tree printing function."""
    spacing = ""
    left =  "["
    right = "]"
    
    # Base case: we've reached a Node
    if isinstance(node, Node):
        # print (spacing + "Predict", node.predictions)
        print(feature_val+left + parent_feature + ', leaf'   + right + ':  [' + str(node.predictions) +right)

        return

    # Print the question at this node
    print(feature_val+ left + parent_feature + ', feature='  +  str(node.feature) + right)
    feature_val += '  '
    # Call this function recursively on the true branch
    #print (feature_val + '--> True:')
    print_tree(node.true_branch, parent_feature = node.feature , feature_val = feature_val)

    # Call this function recursively on the false branch
    #print (feature_val + '--> False:')
    print_tree(node.false_branch, parent_feature = node.feature , feature_val = feature_val )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return 
    
    
    
print_tree(tree_entropy, depth=0, parent_feature='ROOT', feature_val='')