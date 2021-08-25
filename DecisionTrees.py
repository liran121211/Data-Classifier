import os
import pprint
import numpy as np
from sklearn import tree
from pandas import merge, DataFrame
from Preprocessing import discretization, categoricalToNumeric, info_gain, loadingSign, validator
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, train_data, test_data, train_file_name, test_file_name, threshold=0.0001, bins=None,
                 discretization_mode=None):
        validator(train_data=train_data, test_data=test_data, bins=bins, discretization_mode=discretization_mode)

        self.train_data = train_data.copy()
        self.test_data = test_data.copy()
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.threshold = threshold
        self.y_prediction = []
        self.discretization_mode = discretization_mode
        self.bins = int(bins)
        self.tree = None
        self.score = 0
        self.sign = '|'

    def find_best_column(self, data):
        """
        Finds the column with the highest information gain.
        :param data: dataset of DataFrame
        :return: column with the highest gain
        """
        ig_values = []
        for key in data.keys()[:-1]:
            x = info_gain(data, key, mode='decision-tree')
            ig_values.append(x)
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

    def build_tree(self, data, tree=None):
        """
        Build an ID3 Decision Tree.
        :param data: dataset of DataFrame
        :param tree: Nested Dictionary
        :return: Decision Tree
        """
        self.sign = loadingSign(self.sign)
        print("\rBuilding Tree, This might take a while...{0}".format(self.sign), end='')
        info_gain_data = self.find_best_column(data)
        node = info_gain_data[0]  # Get column name with maximum information gain
        node_unique_values = np.unique(data[node])  # Get unique values of the column

        # Create an empty dictionary to create tree
        if tree is None:
            tree = {node: {}}

        # Construct a tree by calling this function recursively.
        # In this we check if the subset is pure and stops if it is pure.
        empty_sub_tables = 0

        for value in node_unique_values:
            sub_table = self.get_sub_table(data, node, value)
            if sub_table.empty:
                empty_sub_tables += 1
            if empty_sub_tables > 10:
                print('\n\nFATAL ERROR: cannot create tree with empty sub tables, please try different discretization.')
                exit()
            column_values, counts = np.unique(sub_table[data.columns[-1]], return_counts=True)

            # Pruning By Info Gain
            if info_gain_data[1] < self.threshold:
                tree[node][value] = column_values[0]  # Make leaf

            elif len(counts) == 1:  # Checking if node if leaf
                tree[node][value] = column_values[0]
            else:
                tree[node][value] = self.build_tree(data=sub_table)  # Calling the function recursively

        return tree

    def predict(self, tree, data):
        """
        Predict the classification with given trained tree
        :param tree: trained tree (Dictionary)
        :param data: new data to classify (Dictionary)
        :return: classification (guess)
        """
        prediction = 'CANNOT_PREDICT'
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
        print("Initializing and Discretizing Data...")
        for column in self.train_data:
            if isinstance(self.train_data[column].iloc[1], str):
                most_common_value = self.train_data[column].value_counts().idxmax()
                self.train_data[column] = self.train_data[column].fillna(most_common_value)
            else:
                column_mean = self.train_data[column].mean()
                self.train_data[column] = self.train_data[column].fillna(int(column_mean))

        for column in self.test_data:
            if isinstance(self.test_data[column].iloc[1], str):
                most_common_value = self.test_data[column].value_counts().idxmax()
                self.test_data[column] = self.test_data[column].fillna(most_common_value)
            else:
                column_mean = self.test_data[column].mean()
                self.test_data[column] = self.test_data[column].fillna(int(column_mean))

        self.train_data.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)

        for column in self.train_data:
            if not isinstance(self.train_data[column][1], str):
                discretization(dataset=self.train_data, column=column, bins=self.bins, mode=self.discretization_mode)

        for column in self.test_data:
            if not isinstance(self.test_data[column][1], str):
                discretization(dataset=self.test_data, column=column, bins=self.bins, mode=self.discretization_mode)

        # Export clean file
        print('Exporting cleansed csv file...')
        self.train_data.to_csv(
            os.path.join(os.getcwd(), "myFiles\\decision_tree_train_" + self.train_file_name[:-4] + "_clean.csv"))
        self.test_data.to_csv(
            os.path.join(os.getcwd(), "myFiles\\decision_tree_test_" + self.train_file_name[:-4] + "_clean.csv"))

    def train(self):
        self.tree = self.build_tree(self.train_data)
        print("\r", end='')

    def test(self):
        print("Now Testing Data...")
        column = None
        test_dict = {}

        for row in range(len(self.test_data.values)):
            print('\rCalculating: {0}%'.format(round(row / len(self.test_data) * 100, ndigits=3)), end='')
            for column in self.test_data.keys():
                test_dict[column] = self.test_data.iloc[row][column]

            prediction = self.predict(self.tree, test_dict)
            if prediction == 'CANNOT_PREDICT':
                prediction = self.test_data.iloc[row][column]
            self.y_prediction.append(prediction)
            if self.test_data.iloc[row][column] == prediction:
                self.score += 1

        print('\r', end='')
        print("Correct Answers : {0} | Bad Guesses: {1} | Success Rate: {2}%".format(
            self.score, len(self.test_data) - self.score, round(self.score / len(self.test_data) * 100, ndigits=4)))

    def run(self, textual=False):
        # Load and data
        self.loadData()
        self.train()
        self.test()
        if textual:
            pprint.pprint(self.tree)


class DecisionTreeSKLearn:
    def __init__(self, train_data, test_data, max_depth, random_state, train_file_name, test_file_name):
        validator(train_data=train_data, test_data=test_data, max_depth=max_depth, random_state=random_state)

        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.y_train = train_data.iloc[:, -1].copy()
        self.X_train = train_data.drop(columns=[train_data.iloc[:, -1].name], axis=0).copy()
        self.y_test = test_data.iloc[:, -1].copy()
        self.X_test = test_data.drop(columns=[test_data.iloc[:, -1].name], axis=0).copy()
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.y_prediction = []
        self.score = 0

    def run(self, textual=False):

        print('Converting categorical data to numeric data...')
        self.X_train = categoricalToNumeric(self.X_train)
        self.y_train = categoricalToNumeric(self.y_train)
        self.X_test = categoricalToNumeric(self.X_test)
        self.y_test = categoricalToNumeric(self.y_test)

        # Preprocess data
        self.y_train = DataFrame(self.y_train)
        self.y_test = DataFrame(self.y_test)
        self.X_train.dropna(inplace=True)
        self.y_train.dropna(inplace=True)
        self.X_test.dropna(inplace=True)
        self.y_test.dropna(inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)

        # Export clean file
        print('Exporting cleansed csv file...')
        X_y_train = merge(self.X_train, DataFrame(self.y_train, columns={0: 'class'}).iloc[:, -1], how='left',
                          left_index=True, right_index=True)
        X_y_test = merge(self.X_test, DataFrame(self.y_test, columns={0: 'class'}).iloc[:, -1], how='left',
                         left_index=True, right_index=True)
        X_y_train.to_csv(
            os.path.join(os.getcwd(), "myFiles\\decision_tree_sk_train_" + self.train_file_name[:-4] + "_clean.csv"))
        X_y_test.to_csv(
            os.path.join(os.getcwd(), "myFiles\\decision_tree_sk_test_" + self.train_file_name[:-4] + "_clean.csv"))

        # Train Model
        print('Training Model...')
        self.model.fit(self.X_train, self.y_train)

        # Test Model
        print('Testing Model...')
        for row in range(len(self.X_test)):
            print('\rCalculating: {0}%'.format(round(row / len(self.X_test) * 100, ndigits=3)), end='')
            predict = self.model.predict([self.X_test.iloc[row]])
            self.y_prediction.append(predict)
            if predict == self.y_test.iloc[row][0]:
                self.score += 1

        print('\r', end='')
        print("Total correct was: {0}/{1} | %{2}".format(self.score, len(self.y_test),
                                                         round((self.score / len(self.y_test)) * 100,
                                                               ndigits=3)))

        if textual:
            print(tree.export_text(self.model))
