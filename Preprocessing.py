from collections import Counter
from sklearn import preprocessing
from math import log
from numpy import ravel
import pandas as pd
import entropy_based_binning as entropy_binning


def minMax(dataset):  # Min-Max Scaling
    minimum = [min(x) for x in zip(*dataset)]
    maximum = [max(x) for x in zip(*dataset)]
    for row in range(len(dataset)):  # Iterate through rows
        for column in range(len(dataset[0])):  # Iterate through columns
            dataset[row][column] = (dataset[row][column] - minimum[column]) / (maximum[column] - minimum[column])
    return dataset


def zScore(dataset):  # Z-Score Scaling
    transpose_dataset = [list(t) for t in zip(*dataset)]  # transpose dataset matrix
    mean_transpose_dataset = [sum(row) / len(row) for row in transpose_dataset]  # for every feature calculate the mean

    for row in range(len(transpose_dataset)):  # Iterate through rows (features)
        feature_deviance = [((feature - mean_transpose_dataset[row]) ** 2) for feature in
                            transpose_dataset[row]]  # list(feature(i) - mean(i)) ^2
        feature_deviance = ((sum(feature_deviance) * (
                1 / (len(feature_deviance))))) ** 0.5  # sqrt(Σ(feature_deviance * 1/len(feature_deviance)

        for column in range(len(transpose_dataset[0])):  # Iterate through columns (data)
            transpose_dataset[row][column] = (transpose_dataset[row][column] - mean_transpose_dataset[
                row]) / feature_deviance  # calculate Z-Score

    return [list(t) for t in zip(*transpose_dataset)]  # return dataset to original normal shape


def decimalScaling(dataset):
    maxValueLength = len(str(max(
        [max(x) for x in dataset])))  # Iterate through all values in dataset and get the length of the maximum value
    for row in range(len(dataset)):  # Iterate through rows
        for column in range(len(dataset[0])):  # Iterate through columns
            dataset[row][column] = (dataset[row][column] / (10 ** maxValueLength))
    return dataset


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
        values_group = []  # Contain the array values of each bin
        for value in dataset:
            if value >= bins_sizes[i] and value <= bins_sizes[i + 1]:  # check which group the value should get into
                values_group.append(value)
        fixed_width_bins.append(values_group)  # add each value to right bin

    return fixed_width_bins


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

    for bin in range(len(sort_data)):  # Iterate through bins
        for index in range(len(sort_data[bin])):  # Iterate through values of each bin
            current = sort_data[bin]  # Make code more crystal clear
            if (index != min_width) and (index != max_width):  # check if bin boundaries has been reached
                if (abs(current[index] - min_bin_value[bin]) < abs(
                        current[index] - max_bin_value[bin])):  # check if value is close to right/left boundary
                    sort_data[bin][index] = min_bin_value[bin]
                else:
                    sort_data[bin][index] = max_bin_value[bin]
    return sort_data


def categoricalToNumeric(dataset):
    """
    Convert Categorical data into Numeric data.
    :param dataset: dataframe (Pandas).
    :return: Numeric dataset.
    """
    pp = preprocessing.LabelEncoder()

    if isinstance(dataset, pd.Series):
        pp.fit(dataset)
        return pp.transform(dataset)

    elif isinstance(dataset, list):
        dataset = ravel(dataset)
        pp.fit(dataset)
        return pp.transform(dataset)
    else:
        for column in dataset:
            if type(column[1]) is str:
                pp.fit(dataset[column])
                dataset[column] = pp.transform(dataset[column])


def discretization(dataset, column, bins, mode):
    """
    :param dataset: Pandas DataFrame
    :param column: Specific column in the dataset
    :param bins: Amount of bins to separate the values in the columns
    :param mode: Choose equal_width / equal frequency / entropy binning
    :return: Discretization on column
    """
    if mode == 'equal-width':
        try:
            dataset[column] = pd.qcut(x=dataset[column], q=bins, labels=[chr(i) for i in range(ord('A'), ord(chr(65+bins)))])
        except ValueError:
            dataset[column] = pd.qcut(x=dataset[column], q=bins, labels=[chr(i) for i in range(ord('A'), ord(chr(65+bins)))], duplicates='drop')

    elif mode == 'equal-frequency':
        dataset[column] = pd.cut(x=dataset[column], bins=bins, labels=[chr(i) for i in range(ord('A'), ord(chr(65+bins)))])

    elif mode == 'entropy':
        dataset[column] = categoricalToNumeric(dataset[column])
        dataset[column] = entropy_binning.bin_sequence(dataset[column], nbins=bins)
    else:
        print('Warning: No discretization selected, process might take longer than expected...\n')


def count_att(data, column, value):
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


def count_conditional_att(data, features_, f_label, class_, c_label=None):
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


def conditional_entropy(data, features_, f_label, class_, l_base=2):
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
        # each column has attribute (this will calculate each att probability)
        probabilities.append(count_conditional_att(data, features_, f_label, class_, c_label))

    return -sum([x * log(x, l_base) for x in probabilities])  # final conditional entropy calc


def basic_entropy(data, features_, l_base=2):
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
        probabilities.append(
            count_att(data, features_, f_label))  # each column has attribute (this will calculate each att probability)

    return -sum([x * log(x, l_base) for x in probabilities])  # final entropy calc


def info_gain(data, column, l_base=2):
    """
    Calculate the information gain given data and specific column
    :param data: dataset of DataFrame
    :param column: column in the dataset
    :param l_base: log base (default: 2)
    :return: calculated information gain
    """
    class_ = data[data.columns[-1]].name  # last column of the DataFrame
    unique_values = data[column].unique()
    sum_gain = basic_entropy(data, class_)
    for feature_ in unique_values:
        sum_gain += -(
                count_conditional_att(data, column, feature_, class_) *
                conditional_entropy(data, column, feature_, class_, l_base))
    return sum_gain


def entropyDiscretization(dataset, column, initial_split, bins_):
    """
    Supervised Binning, Split the data into groups that has the most information gain.
    :param dataset: Pandas DataFrame.
    :param column: column for binning
    :param initial_split: split column for (x) equal groups.
    :param bins_: choose (x) groups to bin that has the most information gain.
    :return: Binned column by entropy.
    """
    info_gain_dict = {}

    # Preserve original dataset
    temp_dataset = dataset.copy()

    # Save classification column name
    class_name = dataset[dataset.columns[-1]].name

    # Split column into (bins) equal groups
    cat, bins = pd.qcut(temp_dataset[column], q=initial_split, retbins=True, duplicates='drop')

    # Iterate thorough each unique bin value in (bins array) and get new sub_table with values greater than bin value
    for value in range(len(bins)):
        subset = dataset[dataset[column] > bins[value]]
        info_gain_dict[dataset[column].iloc[value]] = info_gain(subset, class_name)

    # Finding bin group with the best info gain (up to max_bins groups)
    new_bins = []
    best_values = Counter(info_gain_dict).most_common(bins_)
    for bin in best_values:
        new_bins.append(bin[0])
    new_bins.sort()

    # Binning the original array with best entropy values
    dataset[column] = pd.cut(dataset[column], new_bins, labels=[chr(i) for i in range(
        ord('A'), ord(chr(65 + len(new_bins) - 1)))]).values.add_categories('other')
    dataset[column] = dataset[column].fillna('other')

    return dataset[column]


def validator(**kwargs):
    """
    Validate parameters according to the function requirements.
    :param kwargs: dictionary of parameter_name: parameter
    :return: error if parameter is not fit else nothing.
    """
    if 'train_data' in kwargs and 'test_data' in kwargs:
        if kwargs['train_data'] is None or kwargs['test_data'] is None:
            raise Exception("no train dataset or test dataset were loaded.")

        if not isinstance(kwargs['train_data'], pd.DataFrame) or not isinstance(kwargs['test_data'], pd.DataFrame):
            raise Exception("train dataset or test dataset is not pandas DataFrame object.")

        if len(kwargs['train_data']) == 0 or len(kwargs['test_data']) == 0:
            raise Exception("train dataset or test dataset is empty.")

    if 'confusion_matrix' in kwargs:
        if len(kwargs['confusion_matrix'].columns) < 2:
            raise Exception('Cannot calculate Accuracy / Precision / Recall / F1-Score with 1D array as Confusion '
                            'Matrix.')
    if 'bin' in kwargs:
        if kwargs['bin'] <= 0:
            raise Exception("Bin value must be greater than 0.")

    if 'threshold' in kwargs:
        if kwargs['threshold'] <= 0:
            raise Exception("Threshold value must be greater than 0.")

    if 'max_depth' in kwargs:
        if kwargs['max_depth'] <= 0:
            raise Exception("Max Depth value must be greater than 0.")

    if 'random_state' in kwargs:
        if kwargs['random_state'] <= 0:
            raise Exception("Random State value must be different than 0.")

    if 'k_neighbors' in kwargs:
        if kwargs['k_neighbors'] <= 0:
            raise Exception("K-Neighbors value must be at least 1.")

    if 'k_means' in kwargs:
        if kwargs['k_means'] <= 0:
            raise Exception("K-Means value must be at least 1.")

    if 'max_iterations' in kwargs:
        if kwargs['max_iterations'] <= 0:
            raise Exception("Max Iterations value must be at least 1.")


def loadingSign(state):
    """
    Loading sign for long process...
    :param state: current character state
    :return: next character state
    """
    if state == '|':
        return '/'
    elif state == '/':
        return '——'
    elif state == '——':
        return "\\"
    elif state == '\\':
        return "|"
