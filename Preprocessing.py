# from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing as pp
import pandas as pd
from matplotlib import pyplot as plt
import random as rnd

y = [[1, 2, 500, 6, 9, 8, 5, 2, 1, 4], [4, 5, 9, 3, 2, 0, 1, 4, 5, 8], [5, 2, 3, 6, 9, 8, 5, 2, 1, 4],
     [5, 6, 9, 0, 3, 2, 5, 6, 9, 7], [4, 8, 5, 0, 3, 6, 9, 5, 2, 0], ]


def minMax(dataset):  # Min-Max Scalling
    minimum = [min(x) for x in zip(*dataset)]
    maximum = [max(x) for x in zip(*dataset)]
    for row in range(len(dataset)):  # Iterate through rows
        for column in range(len(dataset[0])):  # Iterate through colums
            dataset[row][column] = (dataset[row][column] - minimum[column]) / (maximum[column] - minimum[column])
    return dataset


def zScore(dataset):  # Z-Score Scalling
    transpose_dataset = [list(t) for t in zip(*dataset)]  # transpose dataset matrix
    mean_transpose_dataset = [sum(row) / len(row) for row in transpose_dataset]  # for every feature calculate the mean

    for row in range(len(transpose_dataset)):  # Iterate through rows (features)
        feature_deviance = [((feature - mean_transpose_dataset[row]) ** 2) for feature in
                            transpose_dataset[row]]  # list(feature(i) - mean(i)) ^2
        feature_deviance = ((sum(feature_deviance) * (
                1 / (len(feature_deviance))))) ** 0.5  # sqrt(Σ(feature_deviance * 1/len(feature_deviance)

        for column in range(len(transpose_dataset[0])):  # Iterate through colums (data)
            transpose_dataset[row][column] = (transpose_dataset[row][column] - mean_transpose_dataset[
                row]) / feature_deviance  # calculate Z-Score

    return [list(t) for t in zip(*transpose_dataset)]  # reutrn dataset to original normal shape


def decimalScaling(dataset):
    maxValueLength = len(str(max(
        [max(x) for x in dataset])))  # Iterate through all values in dataset and get the length of the maximum value
    for row in range(len(dataset)):  # Iterate through rows
        for column in range(len(dataset[0])):  # Iterate through colums
            dataset[row][column] = (dataset[row][column] / (10 ** maxValueLength))
    return dataset


minMaxMethod = pp.MinMaxScaler().fit_transform(y)
z_scoreMethod = pp.StandardScaler().fit_transform(y)
decimalScaleMethod = pp.Normalizer().fit_transform(y)


# ----------------------------------------------------------------------------------------------------------------------------------------------

# Equal-Frequency Binning
def equalFrequency(dataset, n_bins=2):
    bins_array = []  # final array that contains the grouped values
    datasetLength = len(dataset)  # get length of dataset
    frequency = int(datasetLength / n_bins)  # get num of bins (rounded)

    for bin in range(n_bins):
        bins_array.append([])  # create new bin
        for value in range(frequency * bin, frequency * (bin + 1)):  # Iterate each time: (x, x+frequency)
            if value < datasetLength:
                bins_array[bin].append(dataset[value])
    return bins_array


# Equal-Width Binning
def equalWidth(dataset, n_bins=2):
    width = int((max(dataset) - min(dataset)) / n_bins)
    min_value = min(dataset)
    bins_sizes = []  # array that contains the equally size of each bin
    fixed_width_bins = []  # final array that contains the grouped values

    for i in range(n_bins + 1):  # Iterate through (n_bins + 1) to set the group size of the bin
        bins_sizes.append(min_value + (width * i))

    for i in range(n_bins):
        values_group = []  # Contain the array valeus of each bin
        for value in dataset:
            if value >= bins_sizes[i] and value <= bins_sizes[i + 1]:  # check which group the value should get into
                values_group.append(value)
        fixed_width_bins.append(values_group)  # add each value to right bin

    return fixed_width_bins


# Use pandas Cut function to create Equal-Width Binning
def pandasEqualWidth(dataset, n_bins=2, labels=['Label #1', 'Label #2']):
    return pd.cut(dataset, bins=n_bins, labels=labels)


# Use pandas Cut function to create Equal-Frequency Binning
def pandasEqualFrequency(dataset, quantity=2, labels=['Label #1', 'Label #2', 'Label #3']):
    return pd.qcut(dataset, q=quantity, labels=labels)


data2 = [5, 10, 11, 13, 15, 35, 50, 55, 72, 92, 204, 215]
x = equalWidth(data2, n_bins=3)


# -------------------------------------------------------------------------------------------

# SMA Algorithm
def simpleMovingAverage(dataset, window=2):
    sum_dataset = [0]
    final = []

    for index, value in enumerate(dataset, start=1):
        sum_dataset.append(sum_dataset[index - 1] + value)
        if index >= window:  # Calculate averge of (window) items only if index in array reached (window) items
            calc_average = (sum_dataset[index] - sum_dataset[index - window]) / window
            final.append(calc_average)  # add average value to new list

    return final


# WMA Algorithm
def weighedMovingAverage(dataset, weights, window=2):
    DATA = dataset  # Keep original data untouched
    WEIGHTS = weights  # Keep original weights data untouched
    slices = []  # each slice contain value * weight in size of window
    final = []  # all weighted averages will be stored here
    for i in range(len(DATA)):  # Iterate thorugh all dataset values
        if window <= len(DATA):  # check if slice length is not bigger than array length
            for j in range(window):  # Iterate thorugh values in the window frame length
                slices.append(DATA[j] * WEIGHTS[j])  # Product of data value with weight value
            final.append(sum(slices) / sum(WEIGHTS))  # add final mean of (window) frame length
            DATA.pop(0)  # pop first value
            slices = []  # Clear array for next window slice

    return final


# EMA Algorithm
def exponentialMovingAverage(dataset, window=2):
    DATA = dataset
    final = []

    simpleAverage = sum(DATA[:window]) / window  # Calculate (window) values by SMA method
    alpha = float(2 / (1 + window))  # Smoothing (pre-definded function)
    final.append(round(simpleAverage, 3))

    # EMA(today) = ( (Price(today) - EMA(yesterday) ) x alpha ) + EMA(yesterday)
    final.append(round(((DATA[window] - simpleAverage) * alpha) + simpleAverage, 3))

    # Calculate the rest of EMA values
    counter = 1
    for i in DATA[window + 1:]:
        calc = ((i - final[counter]) * alpha) + final[counter]
        counter += 1
        final.append(round(calc, 3))

    return final


# Use pandas to calculate Simple Moving Average
def pandaSMA(dataset, window=2):
    return pd.DataFrame(dataset).rolling(window).mean()


# Use pandas to calculate Weighted Moving Average
def pandaWMA(dataset, weights, window=2):
    weights = np.array(weights)
    return pd.DataFrame(dataset).rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)


# Use pandas to calculate Exponential Moving Average
def pandaEMA(dataset, window=2):
    DATA = pd.DataFrame(dataset)
    final = DATA
    calc_sma = DATA.rolling(window).mean()
    final.iloc[0:window] = calc_sma[0:window]
    return np.round(pd.DataFrame(final).ewm(span=window, adjust=False).mean(), decimals=3)


data_ma = [22.273, 22.194, 22.085, 22.174, 22.184, 22.134, 22.234, 22.432, 22.244, 22.293, 22.154, 22.393, 22.382,
           22.611,
           23.356, 24.052, 23.753, 23.832, 23.952, 23.634, 23.823, 23.872, 23.654, 23.187, 23.098, 23.326, 22.681,
           23.098,
           22.403, 22.173]
weights = [x for x in range(10)]
print(exponentialMovingAverage(data_ma, 5))
print(pandaEMA(data_ma, 5))

# data to be binned
data2 = [5, 10, 11, 13, 15, 35, 50, 55, 72, 92, 204, 215]


# Smoothing By Bin Means
def binMeans(dataset, interval=2):
    sort_data = sorted(equalFrequency(dataset, interval))  # sort data from low to high
    bin_mean = [sum(i) / len(i) for i in sort_data]  # calculate the mean of each bin
    for bin in range(len(sort_data)):
        for value in range(len(sort_data[bin])):
            sort_data[bin][value] = round(bin_mean[bin])  # for every bin replace value by the mean of the bin
    return sort_data


def binBoundaries(dataset, interval=2):
    sort_data = sorted(equalFrequency(dataset, interval))  # sort data from low to high
    min_bin_value = [min(x) for x in sort_data]  # Get 1D list with minimum value of each bin
    max_bin_value = [max(x) for x in sort_data]  # Get 1D list with maximum value of each bin

    max_width = round(len(dataset) / interval) - 1
    min_width = 0

    for bin in range(len(sort_data)):  # Iteate thorugh bins
        for index in range(len(sort_data[bin])):  # Iterate through values of each bin
            current = sort_data[bin]  # Make code more crystal clear
            if (index != min_width) and (index != max_width):  # check if bin boundaries has been reached
                if (abs(current[index] - min_bin_value[bin]) < abs(
                        current[index] - max_bin_value[bin])):  # check if value is close to right/left boundary
                    sort_data[bin][index] = min_bin_value[bin]
                else:
                    sort_data[bin][index] = max_bin_value[bin]
    return sort_data
