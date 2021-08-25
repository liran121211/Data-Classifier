import pprint
import numpy as np

from Preprocessing import discretization, categoricalToNumeric, info_gain
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, train_data, test_data, threshold=0.01):
        self.train_data = train_data
        self.test_data = test_data
        self.threshold = threshold
        self.prediction_data = []
        self.tree = None
        self.score = 0

    def find_best_column(self, data):
        """
        Finds the column with the highest information gain.
        :param data: dataset of DataFrame
        :return: column with the highest gain
        """
        ig_values = []
        for key in data.keys()[:-1]:
            ig_values.append(info_gain(data, key))
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
        for column in self.train_data:
            if not isinstance(self.train_data[column][1], str):
                discretization(dataset=self.train_data, column=column, bins=5, mode='entropy', max_bins=3)

        for column in self.test_data:
            if not isinstance(self.test_data[column][1], str):
                discretization(dataset=self.test_data, column=column, bins=5, mode='entropy', max_bins=3)

        print("Discretization Completed!")

    def train(self):
        print("Building Tree...")
        self.tree = self.build_tree(self.train_data)
        print("Building Tree Completed!")

    def test(self):
        print("Now Testing Data...")
        column = None
        test_dict = {}

        for row in range(len(self.test_data.values)):
            for column in self.test_data.keys():
                test_dict[column] = self.test_data.iloc[row][column]

            prediction = self.predict(self.tree, test_dict)
            self.prediction_data.append(prediction)
            if self.test_data.iloc[row][column] == prediction:
                self.score += 1

        print("Testing Data Completed!")
        print("Correct Answers : {0} | Bad Guesses: {1} | Success Rate: {2}%".format(
            self.score, len(self.test_data) - self.score, round(self.score / len(self.test_data) * 100, ndigits=4)))

    def run(self):
        # Load and data
        self.loadData()
        self.train()
        self.test()
        pprint.pprint(self.tree)


class DecisionTreeSKLearn:
    def __init__(self, train_data, test_data, max_depth, random_state=0):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.y_train = train_data.iloc[:, -1]
        self.X_train = train_data.drop(columns=[train_data.iloc[:, -1].name], axis=0)
        self.y_test = test_data.iloc[:, -1]
        self.X_test = test_data.drop(columns=[test_data.iloc[:, -1].name], axis=0)
        self.y_prediction = []
        self.score = 0

    def run(self):

        # Preprocess data
        print('Converting categorical data to numeric data...')
        categoricalToNumeric(self.X_train)
        categoricalToNumeric(self.y_train)
        categoricalToNumeric(self.X_test)
        categoricalToNumeric(self.y_test)
        print('All data successfully converted!')

        # Train Model
        print('Training Model...')
        self.model.fit(self.X_train, self.y_train)

        # Test Model
        print('Testing Model...')
        for row in range(len(self.X_test)):
            predict = self.model.predict([self.X_test.iloc[row]])
            self.y_prediction.append(predict)
            if predict == self.y_test.iloc[row]:
                self.score += 1

        print("Total correct was: {0}/{1} | %{2}".format(self.score, len(self.y_test),
                                                         round((self.score / len(self.y_test)) * 100,
                                                               ndigits=3)))

# train = pd.read_csv('adult.csv', delimiter=',')[:25000]
# test = pd.read_csv('adult.csv', delimiter=',')[25001:]
# dt = DecisionTreeSKLearn(train_data=train, test_data=test, max_depth=2, random_state=4)
# dt.run()

# Start Model
# train_data = pd.read_csv('adult.csv', delimiter=',')[:25000]
# test_data = pd.read_csv('adult.csv', delimiter=',')[25001:]
# run(train_data, test_data)
