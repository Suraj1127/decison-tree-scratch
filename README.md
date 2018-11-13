# Decision Tree Scratch Implementation
Implementation of decision tree from scratch for classification problems.

## Short Description
Here, decision tree machine learning algorithm is implemented from scratch.
###Features
1. We can do classification using this decision tree.
1. Both Gini impurity and cross-entropy impurity can be used.
2. Compatible with both numerical as well as categorical data.
3. Missing data are handled during training.
4. Missing features in input are handled by taking it as mode of the training data distribution.

## Requirements
We recommend using Python 3 and the implementation here is also done in Python 3 environment.
## Dependencies
- Numpy
- Pandas
- Sklearn (For splitting utility function)

## Installation
 Install Pandas, Numpy and Sklearn by executing the following commands in the terminal.
```
$ sudo apt-get install python-pip  
$ sudo pip install numpy scipy
$ sudo pip install pandas
$ sudo pip install scikit-learn
```

## Instructions
* Run `./decision_tree.py` with python3 as the interpreter(shebang would take care in our file) and the system would
print out accuracy in titanic validation data.
* Employ your own data and start modelling.  The code is well documented.

## Further Enhancements
* We can implement classification task from this decision tree.  We can have regression implemented in the coming days.
* Selection of separator is done brute force.  We can sort the values first and then select separator using quartiles, percentiles or any such measure.