import pprint
from math import log
import pandas as pd
import numpy as np
from Preprocessing import discretization


class DecisionTree:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.tree = None

    def count_att(self, data, column, value):
        """
        :param data: Pandas DataFrame
        :param column: specific column in the dataset
        :param value: which value in the column should be counted
        :return: probability of (value) to show in (column), included Laplacian correction
        """
        dataset_len = len(data)
        try:  # if (value) not been found then return laplacian calculation to preserve the probability
            p = data[column].value_counts()[value] / dataset_len
            if p == 0:
                p = 1 / (dataset_len + len(data[column].value_counts()))
            return p
        except KeyError:
            return 1 / (dataset_len + len(data[column].value_counts()))

    def count_conditional_att(self, data, features_, f_label, class_, c_label=None):
        """
        :param data: Pandas DataFrame
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
                p = pd.crosstab(data[class_], data[features_], normalize='columns')[f_label][c_label]
                if p == 0:
                    p = 1
                return p
            except KeyError:
                return 1
        else:
            try:  # if (f_label) not been found then return 1 to preserve the probability
                p = pd.crosstab(data[features_], class_, normalize='columns').transpose()[f_label][0]
                if p == 0:
                    p = 1
                return p
            except KeyError:
                return 1

    def conditional_entropy(self, data, features_, f_label, class_, l_base=2):
        """
        Calculate the conditional entropy (Y/X)= −∑p(x,y) · log(p(y/x))
        :param data: dataset of DataFrame
        :param features_: column in the dataset
        :param f_label: attribute (label) in the features_ column
        :param class_: (class) which represent the classification column
        :param l_base: which log the entropy will be calculated
        :return: conditional entropy calculation
        """
        probabilities = []  # Probabilities list
        column_labels = data[class_].unique()  # extract unique attributes to list
        for c_label in column_labels:
            probabilities.append(
                self.count_conditional_att(data, features_, f_label, class_,
                                           c_label))  # each column has attribute (this will calculate each att probability)

        return -sum([x * log(x, l_base) for x in probabilities])  # final conditional entropy calc

    def basic_entropy(self, data, features_, l_base=2):
        """
        Calculate the entropy (X)= −∑p(x) · log(p(x))
        :param data: dataset of DataFrame
        :param features_: attribute in the column of the dataset
        :param l_base: which log base, the entropy will be calculated with.
        :return: basic entropy calculation
        """
        probabilities = []  # Probabilities list
        column_labels = data[features_].unique()  # extract unique attributes to list
        for f_label in column_labels:
            probabilities.append(self.count_att(data, features_,f_label))  # each column has attribute (this will calculate each att probability)

        return -sum([x * log(x, l_base) for x in probabilities])  # final entropy calc

    def info_gain(self, data, column, l_base=2):
        """
        Calculate the information gain given data and specific column
        :param data: dataset of DataFrame
        :param column: column in the dataset
        :param l_base: log base (default: 2)
        :return: calculated information gain
        """
        class_ = data[data.columns[-1]].name  # last column of the DataFrame
        unique_values = data[column].unique()
        sum_gain = self.basic_entropy(data, class_)
        for feature_ in unique_values:
            sum_gain += -(
                    self.count_conditional_att(data, column, feature_, class_) *
                    self.conditional_entropy(data, column, feature_, class_, l_base))
        return sum_gain

    def find_best_column(self, data):
        """
        Finds the column with the highest information gain.
        :param data: dataset of DataFrame
        :return: column with the highest gain
        """
        ig_values = []
        for key in data.keys()[:-1]:
            ig_values.append(self.info_gain(data, key))
        return data.keys()[:-1][np.argmax(ig_values)], max(ig_values)

    def get_sub_table(self, data, column, label):
        """
        Create a sub table with column that contains only "label" values
        :param data: dataset of DataFrame
        :param column: Column name
        :param label: Unique value of the column
        :return:
        """
        temp_table = data[data[column] == label].reset_index(drop=True)

        # check if table contain more than 100 records
        if len(temp_table) > 100:
            return data[data[column] == label].reset_index(drop=True)[:100]
        else:
            return data[data[column] == label].reset_index(drop=True)

    def build_tree(self, data, gain_threshold=0.000000001, tree=None):
        """
        Build an ID3 Decision Tree.
        :param data: dataset of DataFrame
        :param tree: Nested Dictionary
        :param gain_threshold: if below threshold then make leaf
        :return: Decision Tree
        """
        info_gain_data = self.find_best_column(data)
        node = info_gain_data[0]  # Get column name with maximum information gain
        node_unique_values = np.unique(data[node])  # Get unique values of the column

        # Create an empty dictionary to create tree
        if tree is None:
            tree = {node: {}}

        # Construct a tree by calling this function recursively.
        # In this we check if the subset is pure and stops if it is pure.

        for value in node_unique_values:
            sub_table = self.get_sub_table(data, node, value)
            column_values, counts = np.unique(sub_table[data.columns[-1]], return_counts=True)

            print(sub_table)

            # Pruning By Info Gain
            if info_gain_data[1] < gain_threshold:
                tree[node][value] = column_values[0]  # Make leaf

            elif len(counts) == 1:  # Checking if node if leaf
                tree[node][value] = column_values[0]
            else:
                tree[node][value] = self.build_tree(
                    data=sub_table, gain_threshold=gain_threshold)  # Calling the function recursively

        return tree

    def predict(self, tree, data):
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
                prediction = self.predict(tree, data)
            else:  # Return classification value
                prediction = tree
                break

        return prediction

    def loadData(self):
        """
        Initialize train and test data, drop all NaN values, make discretization on each of the Numeric columns
        :return: processed train and test data
        """
        self.train_data.dropna(inplace=True)
        self.train_data.reset_index(drop=True, inplace=True)
        self.test_data.dropna(inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)

        print("Initializing and Discretizing Data...")
        discretization(self.train_data, 'age', 3, "equal-frequency", ['Young', 'Old', 'Elder'])
        discretization(self.train_data, 'balance', 3, "equal-frequency", ["Low", "Average", "High"])
        discretization(self.train_data, 'campaign', 3, "equal-frequency", ["New_Lead", "Ongoing_Lead", "Veteran_Lead"])
        discretization(self.train_data, 'day', 3, "equal-frequency",
                       ["Early_This_Month", "Middle_Of_Month", "Lately_This_Month"])
        discretization(self.train_data, 'duration', 3, "equal-frequency", ["Short_Call", "Average_Call", "Long_Call"])
        discretization(self.train_data, 'previous', 3, "equal-frequency", ["New_Lead", "Ongoing_Lead", "Hot_Lead"])

        discretization(self.test_data, 'age', 3, "equal-frequency", ['Young', 'Old', 'Elder'])
        discretization(self.test_data, 'balance', 3, "equal-frequency", ["Low", "Average", "High"])
        discretization(self.test_data, 'campaign', 3, "equal-frequency", ["New_Lead", "Ongoing_Lead", "Veteran_Lead"])
        discretization(self.test_data, 'day', 3, "equal-frequency",
                       ["Early_This_Month", "Middle_Of_Month", "Lately_This_Month"])
        discretization(self.test_data, 'duration', 3, "equal-frequency", ["Short_Call", "Average_Call", "Long_Call"])
        discretization(self.test_data, 'previous', 3, "equal-frequency", ["New_Lead", "Ongoing_Lead", "Hot_Lead"])
        print("Initialized and Discretization Data Completed!")

    def train(self):
        print("Building Tree...")
        self.tree = self.build_tree(self.train_data, gain_threshold=0.01)
        print("Building Tree Completed!")

    def test(self):
        print("Now Testing Data...")
        column = None
        match = 0
        test_dict = {}

        for row in range(len(self.test_data.values)):
            for column in self.test_data.keys():
                test_dict[column] = self.test_data.iloc[row][column]

            if self.test_data.iloc[row][column] == self.predict(self.tree, test_dict):
                match += 1

        print("Testing Data Completed!")
        print("Correct Answers : {0} | Bad Guesses: {1} | Success Rate: {2}%".format(
            match, len(self.test_data) - match, round(match / len(self.test_data) * 100, ndigits=4)))


def run():
    # Load and data
    train_data = pd.read_csv('train.csv', delimiter=',')
    test_data = pd.read_csv('test.csv', delimiter=',')

    decision_tree = DecisionTree(train_data, test_data)
    decision_tree.loadData()
    decision_tree.train()
    decision_tree.test()


# Start Model
run()
