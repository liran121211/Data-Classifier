import pprint
from math import log
import pandas as pd
import numpy as np


def discretization(dataset, column, bins, mode, labels=None):
    """
    :param dataset: Pandas DataFrame
    :param column: specific column in the dataset
    :param bins: amount of bins to separate the values in the columns
    :param labels: rename the bins into specific name
    :param mode: choose equal_width / equal frequency binning
    :return: discretization on column
    """
    if mode == 'equal-width':
        dataset[column] = pd.qcut(dataset[column], q=bins, labels=labels)

    if mode == 'equal-frequency':
        dataset[column] = pd.cut(dataset[column], bins=bins, labels=labels)

    else:
        raise NameError("Mode does not exist!")


def count_att(dataset, column, value):
    """
    :param dataset: Pandas DataFrame
    :param column: specific column in the dataset
    :param value: which value in the column should be counted
    :return: probability of (value) to show in (column), included Laplacian correction
    """
    dataset_len = len(dataset)
    try:  # if (value) not been found then return laplacian calculation to preserve the probability
        p = dataset[column].value_counts()[value] / dataset_len
        if p == 0:
            p = 1 / (dataset_len + len(dataset[column].value_counts()))
        return p
    except KeyError:
        return 1 / (dataset_len + len(dataset[column].value_counts()))


def count_conditional_att(dataset, features_, f_label, class_, c_label=None):
    """
    :param dataset: Pandas DataFrame
    :param features_: First selected column
    :param f_label: Second selected column
    :param class_: Independent value of column1
    :param c_label: Dependent value of column2
    :return: conditional probability of Y when X already occurred.
    P(A|B)=P(B|A)P(A)/P(B)
    P(class|features)=P(features|class) * P(class) / P(features)
    """
    if c_label is not None:
        try:  # if (f_label) not been found then return 1 to preserve the probability
            p = pd.crosstab(dataset[class_], dataset[features_], normalize='columns')[f_label][c_label]
            if p == 0:
                p = 1
            return p
        except KeyError:
            return 1
    else:
        try:  # if (f_label) not been found then return 1 to preserve the probability
            p = pd.crosstab(dataset[features_], class_, normalize='columns').transpose()[f_label][0]
            if p == 0:
                p = 1
            return p
        except KeyError:
            return 1


def conditional_entropy(dataset, features_, f_label, class_, l_base=2):
    """
    Calculate the conditional entropy (Y/X)= −∑p(x,y) · log(p(y/x))
    :param dataset: dataset of DataFrame
    :param features_: column in the dataset
    :param f_label: attribute (label) in the features_ column
    :param class_: (class) which represent the classification column
    :param l_base: which log the entropy will be calculated
    :return: conditional entropy calculation
    """
    probabilities = []  # Probabilities list
    column_labels = dataset[class_].unique()  # extract unique attributes to list
    for c_label in column_labels:
        probabilities.append(
            count_conditional_att(dataset, features_, f_label, class_,
                                  c_label))  # each column has attribute (this will calculate each att probability)

    return -sum([x * log(x, l_base) for x in probabilities])  # final conditional entropy calc


def basic_entropy(dataset, features_, l_base=2):
    """
    Calculate the entropy (X)= −∑p(x) · log(p(x))
    :param dataset: dataset of DataFrame
    :param features_: attribute in the column of the dataset
    :param l_base: which log base, the entropy will be calculated with.
    :return: basic entropy calculation
    """
    probabilities = []  # Probabilities list
    column_labels = dataset[features_].unique()  # extract unique attributes to list
    for f_label in column_labels:
        probabilities.append(count_att(dataset, features_,
                                       f_label))  # each column has attribute (this will calculate each att probability)

    return -sum([x * log(x, l_base) for x in probabilities])  # final entropy calc


def info_gain(dataset, column, l_base=2):
    """
    Calculate the information gain given data and specific column
    :param dataset: dataset of DataFrame
    :param column: column in the dataset
    :param l_base: log base (default: 2)
    :return: calculated information gain
    """
    class_ = dataset[dataset.columns[-1]].name  # last column of the DataFrame
    unique_values = dataset[column].unique()
    sum_gain = basic_entropy(dataset, class_)
    for feature_ in unique_values:
        sum_gain += -(count_conditional_att(dataset, column, feature_, class_) * conditional_entropy(dataset, column,
                                                                                                     feature_, class_,
                                                                                                     l_base))
    return sum_gain


def find_best_column(dataset):
    """
    Finds the column with the highest information gain.
    :param dataset: dataset of DataFrame
    :return: column with the highest gain
    """
    ig_values = []
    for key in dataset.keys()[:-1]:
        ig_values.append(info_gain(dataset, key))
    return dataset.keys()[:-1][np.argmax(ig_values)], max(ig_values)


def get_sub_table(dataset, column, label):
    """
    Create a sub table with column that contains only "label" values
    :param dataset: dataset of DataFrame
    :param column: Column name
    :param label: Unique value of the column
    :return:
    """
    temp_table = dataset[dataset[column] == label].reset_index(drop=True)

    # check if table contain more than 100 records
    if len(temp_table) > 100:
        return dataset[dataset[column] == label].reset_index(drop=True)[:100]
    else:
        return dataset[dataset[column] == label].reset_index(drop=True)


def build_tree(dataset, gain_threshold=0.000000001, tree=None):
    """
    Build an ID3 Decision Tree.
    :param dataset: dataset of DataFrame
    :param tree: Nested Dictionary
    :param gain_threshold: if below threshold then make leaf
    :return: Decision Tree
    """
    info_gain_data = find_best_column(dataset)
    node = info_gain_data[0]  # Get column name with maximum information gain
    node_unique_values = np.unique(dataset[node])  # Get unique values of the column

    # Create an empty dictionary to create tree
    if tree is None:
        tree = {node: {}}

    # Construct a tree by calling this function recursively.
    # In this we check if the subset is pure and stops if it is pure.

    for value in node_unique_values:
        sub_table = get_sub_table(dataset, node, value)
        column_values, counts = np.unique(sub_table[dataset.columns[-1]], return_counts=True)

        print(sub_table)

        # Pruning By Info Gain
        if info_gain_data[1] < gain_threshold:
            tree[node][value] = column_values[0]  # Make leaf

        elif len(counts) == 1:  # Checking if node if leaf
            tree[node][value] = column_values[0]
        else:
            tree[node][value] = build_tree(dataset=sub_table,
                                           gain_threshold=gain_threshold)  # Calling the function recursively
    return tree


def predict(tree, data):
    """
    Predict the classification with given trained tree
    :param tree: trained tree (Dictionary)
    :param data: new data to classify (Dictionary)
    :return: classification (guess)
    """
    prediction = 0
    for node in tree.keys():
        try:
            node_value = data[node]
            tree = tree[node][node_value]
        except KeyError:
            continue

        if type(tree) is dict:  # Continue to travel in the tree
            prediction = predict(tree, data)
        else:  # Return classification value
            prediction = tree
            break

    return prediction


# --------------------------------------------Dummy Data--------------------------------------------------------
# day = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
outlook = ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
           'Overcast', 'Overcast', 'Rain'] * 1
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'] * 1
humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High',
            'Normal', 'High'] * 1
wind = ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
        'Weak', 'Strong'] * 1
decision = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'] * 1
DATA = pd.DataFrame([outlook, temp, humidity, wind, decision])
DATA = DATA.transpose()
DATA.rename(columns={0: "outlook", 1: "temp", 2: "humidity", 3: "wind", 4: "decision"}, inplace=True)


# --------------------------------------------Dummy Data--------------------------------------------------------

def init():
    """
    Initialize train and test data, drop all NaN values, make discretization on each of the Numeric columns
    :return: train and test DataFrames
    """
    train_data = pd.read_csv('train.csv', delimiter=',')
    test_data = pd.read_csv('test.csv', delimiter=',')

    train_data.dropna(inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    test_data.dropna(inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    discretization(train_data, 'age', 3, "equal-frequency", ['Young', 'Old', 'Elder'])
    discretization(train_data, 'balance', 3, "equal-frequency", ["Low", "Average", "High"])
    discretization(train_data, 'campaign', 3, "equal-frequency", ["New_Lead", "Ongoing_Lead", "Veteran_Lead"])
    discretization(train_data, 'day', 3, "equal-frequency",
                   ["Early_This_Month", "Middle_Of_Month", "Lately_This_Month"])
    discretization(train_data, 'duration', 3, "equal-frequency", ["Short_Call", "Average_Call", "Long_Call"])
    discretization(train_data, 'previous', 3, "equal-frequency", ["New_Lead", "Ongoing_Lead", "Hot_Lead"])

    discretization(test_data, 'age', 3, "equal-frequency", ['Young', 'Old', 'Elder'])
    discretization(test_data, 'balance', 3, "equal-frequency", ["Low", "Average", "High"])
    discretization(test_data, 'campaign', 3, "equal-frequency", ["New_Lead", "Ongoing_Lead", "Veteran_Lead"])
    discretization(test_data, 'day', 3, "equal-frequency", ["Early_This_Month", "Middle_Of_Month", "Lately_This_Month"])
    discretization(test_data, 'duration', 3, "equal-frequency", ["Short_Call", "Average_Call", "Long_Call"])
    discretization(test_data, 'previous', 3, "equal-frequency", ["New_Lead", "Ongoing_Lead", "Hot_Lead"])

    return train_data, test_data


def train():
    print("Initializing and Discretizing Data...")
    train_data = init()[0]
    print("Initialized and Discretization Data Completed!")

    print("Building Tree...")
    tree_ = build_tree(dataset=train_data,gain_threshold=0.01)
    print("Building Tree Completed!")
    return tree_


def test(tree):
    key = None
    correct = 0
    not_correct = 0
    print("Now Testing Data...")
    test_dict = {}
    test_data = init()[1]

    for row in range(len(test_data.values)):
        for key in test_data.keys():
            test_dict[key] = test_data.iloc[row][key]
        if predict(tree, test_dict) == test_data.iloc[row][key]:
            correct += 1
        else:
            not_correct += 1

    print("Testing Data Completed!")
    print("Correct Answers : {0} | Bad Guesses: {1} | Success Rate: {2}%".format(correct, not_correct,
                                                                                 round(correct / len(test_data) * 100,
                                                                                       ndigits=4)))


x = train()
test(x)