import os
from math import sqrt
from Preprocessing import categoricalToNumeric, validator
from pandas import merge, DataFrame

class KNN:
    def __init__(self, train_file, test_file, k_neighbors, train_file_name, test_file_name):
        validator(train_data=train_file, test_data=test_file, k_neighbors=k_neighbors)
        self.k_neighbors = k_neighbors
        self.train_data = train_file
        self.test_data = test_file
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.y_prediction = []
        self.score = 0

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

    def getNeighbors(self, train_dataset, test_row, k_neighbors):
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
        :param test_row: row to be classified.
        :return: Classified point.
        """
        neighbors = self.getNeighbors(self.train_data, test_row, self.k_neighbors)
        class_column = [row[-1] for row in neighbors]
        predict = max(set(class_column), key=class_column.count)  # return the max value with the most occurrences
        return predict

    def loadData(self):
        """
        Load and preprocess the datasets.
        :return: processed datasets.
        """
        print("Initializing Data...")
        for column in self.train_data:
            if (isinstance(self.train_data[column].iloc[1], str)):
                most_common_value = self.train_data[column].value_counts().idxmax()
                self.train_data[column] = self.train_data[column].fillna(most_common_value)
            else:
                column_mean = self.train_data[column].mean()
                self.train_data[column] = self.train_data[column].fillna(int(column_mean))

        for column in self.test_data:
            if (isinstance(self.test_data[column].iloc[1], str)):
                most_common_value = self.test_data[column].value_counts().idxmax()
                self.test_data[column] = self.test_data[column].fillna(most_common_value)
            else:
                column_mean = self.test_data[column].mean()
                self.test_data[column] = self.test_data[column].fillna(int(column_mean))

        self.train_data.reset_index(drop=True, inplace=True)
        self.test_data.reset_index(drop=True, inplace=True)

        self.train_data = categoricalToNumeric(self.train_data)
        self.test_data = categoricalToNumeric(self.test_data)

        # Export clean file
        print('Exporting cleansed csv file...')
        self.train_data.to_csv(os.path.join(os.getcwd(), "KNN_train_" + self.train_file_name[:-4] + "_clean.csv"))
        self.test_data.to_csv(os.path.join(os.getcwd(), "KNN__test_" + self.train_file_name[:-4] + "_clean.csv"))
        print("Initialized Data Completed!")

    def prediction(self):
        """
        Start prediction with KNN algorithm
        :return: Success Prediction Rate.
        """
        print('Prediction Started!')
        for row in range(len(self.test_data)):  # Classify each row in the test with all rows in the train
            guess = self.classifyPoint(self.test_data.iloc[row])
            self.y_prediction.append(guess)
            print('\rCalculating: {0}%'.format(round(row / len(self.test_data) * 100, ndigits=3)), end='')
            if self.test_data.iloc[row][-1] == guess:
                self.score += 1

        print('\r', end='')
        print('Success Rate {0}'.format((self.score / len(self.test_data)) * 100))

    def run(self):
        self.loadData()
        self.prediction()
