import pandas as pd

from Evaluator import confusionMatrix
from Preprocessing import discretization, categoricalToNumeric, count_conditional_att, count_att
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.class_col_name = None
        self.class_uniques = []
        self.probabilities = {}
        self.class_probabilities = []
        self.score = 0
        self.prediction_column = []

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
            if not isinstance(self.train_dataset[column][1], str) and column != self.class_col_name:
                discretization(self.train_dataset, column, 5, "equal-frequency",
                               [chr(i) for i in range(ord('A'), ord('E') + 1)])

        for column in self.test_dataset:
            if not isinstance(self.test_dataset[column][1], str) and column != self.class_col_name:
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
                    no_dict[(column, value)] = count_conditional_att(self.train_dataset, column, value,
                                                                          self.class_col_name, self.class_uniques[0])
                    yes_dict[(column, value)] = count_conditional_att(self.train_dataset, column, value,
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

        p_class_n = count_att(self.train_dataset, self.class_col_name, self.class_uniques[0])
        p_class_y = count_att(self.train_dataset, self.class_col_name, self.class_uniques[1])

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
            self.prediction_column.append(answer)
            real_answer = (self.test_dataset.iloc[row][-1])
            if answer == real_answer:
                count_correct += 1

        self.score = count_correct
        print('Testing Completed!')
        print("Total correct was: {0}/{1} | %{2}".format(self.score, len(self.test_dataset),
                                                         round((self.score / len(self.test_dataset)) * 100, ndigits=3)))


def run(train_data, test_data):
    # Load Information
    naive_bayes = NaiveBayes()
    naive_bayes.loadData(train_data, test_data)
    naive_bayes.train()
    naive_bayes.test(naive_bayes.class_probabilities[1], naive_bayes.class_probabilities[0])
    print(confusionMatrix(test[test.columns[-1]], naive_bayes.prediction_column))


class NaiveBayes_SKLearn:
    def __init__(self, train_data, test_data):
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.train_class_col = self.train_dataset[self.train_dataset.columns[-1]]
        self.test_class_col = self.test_dataset[self.test_dataset.columns[-1]]
        self.model = GaussianNB()
        self.score = 0
        self.prediction_column = []

    def run(self):
        success_guess = 0

        # Preprocess data
        categoricalToNumeric(self.train_dataset)
        categoricalToNumeric(self.test_dataset)
        self.train_dataset = self.train_dataset.drop(columns=[self.train_class_col.name], axis=0)
        self.test_dataset = self.test_dataset.drop(columns=[self.test_class_col.name], axis=0)

        # Train Model
        print('Training Model...')
        self.model.fit(self.train_dataset, self.train_class_col)

        # Test Model
        print('Testing Model...')
        for row in range(len(self.test_dataset)):
            prediction = self.model.predict([self.test_dataset.iloc[row]])
            self.prediction_column.append(prediction)
            if prediction == self.train_class_col.iloc[row]:
                success_guess += 1

        print("Total correct was: {0}/{1} | %{2}".format(success_guess, len(test),
                                                         round((success_guess / len(test)) * 100, ndigits=3)))


# Start Model
train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')
#
# nb_sklearn = NaiveBayes_SKLearn(train, test)
# nb_sklearn.run()
#
run(train, test)


