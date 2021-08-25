from math import sqrt

def rowsDistance(row1, row2):
    '''
    Calculate the distance between 2 rows
    The distance is calculated by Euclidean Distance
    The Function sums each column of the 2 rows.
    :param row1: First row.
    :param row2: Second row.
    :return: The square root of the sum (distance).
    '''
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def get_neighbors(train_dataset, test_row, k_neighbors):
    '''
    Calculates the k nearest neighbors
    :param train_dataset: tran dataset
    :param test_row: specific row in the test dataset
    :param k_neighbors: amount of closest neighbors on a specific point
    :return: the nearest neighbors of the point
    '''
    distances = []
    for train_row in train_dataset:
        row_dist = rowsDistance(test_row, train_row)
        distances.append((train_row, row_dist))
        distances.sort(
            key=lambda tup: tup[1])  # ensuring that the second item in the tuple is used in the sorting operation
    neighbors = []

    for i in range(k_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def prediction(train_dataset, test_row, k_neighbors):
    '''
    Classify point to the nearest points.
    :param train_dataset: train data
    :param test_row: specific train row
    :param k_neighbors: amount of closest neighbors on a specific point.
    :return: Classified point.
    '''
    neighbors = get_neighbors(train_dataset, test_row, k_neighbors)
    class_column = [row[-1] for row in neighbors]
    prediction = max(set(class_column), key=class_column.count)  # return the max value with the most occurrences
    return prediction

import pandas as pd
train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')
train.dropna(inplace=True)
train.reset_index(drop=True, inplace=True)
test.dropna(inplace=True)
test.reset_index(drop=True, inplace=True)

print('Expected {0}, Got {1}'.format(test.iloc[10][-1], prediction(train,test.iloc[10],5)))
