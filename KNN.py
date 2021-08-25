from math import sqrt
import pandas as pd

from Preprocessing import categoricalToNumeric

class KNN():
    def __init__(self, k_neighbors):
        self.k_neighbors = k_neighbors
        self.train_data = None
        self.test_data = None

    def rowsDistance(self, row1, row2):
        """
        Calculate the distance between 2 rows
        The distance is calculated by Euclidean Distance
        The Function sums each column of the 2 rows.
        :param row1: First row.
        :param row2: Second row.
        :return: The square root of the sum (distance).
        """
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    def get_neighbors(self, train_dataset, test_row, k_neighbors):
        """
        Calculates the k nearest neighbors
        :param train_dataset: tran dataset
        :param test_row: specific row in the test dataset
        :param k_neighbors: amount of closest neighbors on a specific point
        :return: the nearest neighbors of the point
        """
        distances = []
        for train_row in range(len(train_dataset)):
            row_dist = self.rowsDistance(test_row, train_dataset.iloc[train_row])
            distances.append((train_dataset.iloc[train_row], row_dist))
            distances.sort(
                key=lambda tup: tup[1])  # ensuring that the second item in the tuple is used in the sorting operation
        neighbors = []

        for i in range(k_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def classifyPoint(self, test_row):
        """
        Classify point to the nearest points.
        :param train_dataset: train data
        :param test_row: specific train row
        :param k_neighbors: amount of closest neighbors on a specific point.
        :return: Classified point.
        """
        neighbors = self.get_neighbors(self.train_data, test_row, self.k_neighbors)
        class_column = [row[-1] for row in neighbors]
        predict = max(set(class_column), key=class_column.count)  # return the max value with the most occurrences
        return predict

    def loadData(self, train_file, test_file):
        """
        Load and preprocess the datasets.
        :return: processed datasets.
        """
        self.train_data = train_file
        self.test_data = test_file

        self.train_data.dropna(inplace=True)
        self.train_data.reset_index(drop=True, inplace=True)
        self.test_data.dropna(inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)
        categoricalToNumeric(self.train_data)
        categoricalToNumeric(self.test_data)

    def prediction(self):
        """
        Start prediction with KNN algorithm
        :param train_size: amount of rows in train dataset
        :param test_size: amount of rows in test dataset
        :return: Success Prediction Rate.
        """
        success_guess = 0
        print('Prediction Started!')
        for row in range(len(self.test_data)):  # Classify each row in the test with all rows in the train
            guess = self.classifyPoint(self.test_data.iloc[row])
            print('Actual classification: {0} | Guess Classification: {1}'.format(self.test_data.iloc[row][-1], guess))
            if self.test_data.iloc[row][-1] == guess:
                success_guess += 1

        print('Success Rate {0}'.format((success_guess / len(self.test_data)) * 100))


def run():
    train = pd.read_csv('train.csv', delimiter=',')[:100]
    test = pd.read_csv('test.csv', delimiter=',')
    knn = KNN(k_neighbors=5)
    knn.loadData(train, test)
    knn.prediction()

# Star Model
run()
