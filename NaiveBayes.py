from math import log
from Preprocessing import discretization
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.class_col_name = None
        self.class_uniques = []
        self.probabilities = {}
        self.class_probabilities = []
        self.score = 0

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

    def loadData(self, train_file, test_file):
        """
        Initialize train and test data, drop all NaN values, make discretization on each of the Numeric columns
        :return: train and test DataFrames
        """
        self.train_dataset = train_file
        self.test_dataset = test_file
        self.train_dataset.dropna(inplace=True)
        self.train_dataset.reset_index(drop=True, inplace=True)
        self.test_dataset.dropna(inplace=True)
        self.test_dataset.reset_index(drop=True, inplace=True)
        self.class_col_name = self.train_dataset[self.train_dataset.columns[-1]].name

        for attribute in self.train_dataset[self.train_dataset.columns[-1]].unique():
            self.class_uniques.append(attribute)

        print("Initializing and Discretizing Data...")
        for column in self.train_dataset:
            if not isinstance(self.train_dataset[column][1], str):
                discretization(self.train_dataset, column, 5, "equal-frequency",
                               [chr(i) for i in range(ord('A'), ord('E') + 1)])

        for column in self.test_dataset:
            if not isinstance(self.test_dataset[column][1], str):
                discretization(self.test_dataset, column, 5, "equal-frequency",
                               [chr(i) for i in range(ord('A'), ord('E') + 1)])
        print("Discretization Completed!")

        for column in self.train_dataset:
            self.probabilities[column] = self.train_dataset[column].unique()

        print("Initialized Data Completed!")

    def train(self):
        """
        save all of the conditional probabilities in the dictionary for each of the values 'Age':['A': 0.2685]
        :return: yes and no dictionaries that contain all of the conditional probabilities with 'class' [yes/no] respectively
        """
        print('Training Started...')

        # Calculate Probabilities for each unique (class column) value
        no_dict = {}
        yes_dict = {}

        for column, array in self.probabilities.items():
            for value in array:
                if column.lower() != self.class_col_name:  # if (column) is not last classification column
                    no_dict[(column, value)] = self.count_conditional_att(self.train_dataset, column, value,
                                                                          self.class_col_name, self.class_uniques[0])
                    yes_dict[(column, value)] = self.count_conditional_att(self.train_dataset, column, value,
                                                                           self.class_col_name, self.class_uniques[1])

        self.class_probabilities.append(no_dict)
        self.class_probabilities.append(yes_dict)
        print('Training Completed!')

    def naive_bayes_classifier(self, yes_dict, no_dict, row):
        """
        Get Test dataset, row in the test dataset to be classified, yes/no dictionaries
        :param yes_dict: conditional probabilities that contain calculation with [class: 'yes']
        :param no_dict: conditional probabilities that contain calculation with [class: 'no']
        :param row: row of data to be classified
        :return: classified row with two options: yes/no
        """
        prod_prob_y = 1
        prod_prob_n = 1
        for column in self.train_dataset:
            if column != self.class_col_name:
                try:
                    prod_prob_y *= yes_dict[(column, self.train_dataset[column][row])]
                except KeyError:
                    prod_prob_y *= 1
                try:
                    prod_prob_n *= no_dict[(column, self.train_dataset[column][row])]
                except KeyError:
                    prod_prob_n *= 1

        p_class_n = self.count_att(self.train_dataset, self.class_col_name, self.class_uniques[0])
        p_class_y = self.count_att(self.train_dataset, self.class_col_name, self.class_uniques[1])

        if prod_prob_y * p_class_y > prod_prob_n * p_class_n:
            return self.class_uniques[1]
        else:
            return self.class_uniques[0]

    def test(self, y_dict, n_dict):
        """
        Calculate the success rate of (naive_bayes_classifier) and return the amount of correct predictions
        :param y_dict: conditional probabilities that contain calculation with [class: 'yes']
        :param n_dict: conditional probabilities that contain calculation with [class: 'no']
        :return: amount of correct predictions
        """
        print('Testing Started...')
        count_correct = 0
        for row in range(len(self.test_dataset)):
            answer = self.naive_bayes_classifier(yes_dict=y_dict, no_dict=n_dict, row=row)
            real_answer = (self.test_dataset.iloc[row][-1])
            if answer == real_answer:
                count_correct += 1

        self.score = count_correct
        print('Testing Completed!')
        print("Total correct was: {0}/{1} | %{2}".format(self.score, len(self.test_dataset),round((self.score / len(self.test_dataset)) * 100,ndigits=3)))

def run():
    # Load Information
    train = pd.read_csv('train.csv', delimiter=',')
    test = pd.read_csv('test.csv', delimiter=',')

    naive_bayes = NaiveBayes()
    naive_bayes.loadData(train, test)
    naive_bayes.train()
    naive_bayes.test(naive_bayes.class_probabilities[1], naive_bayes.class_probabilities[0])



# Start Model
run()