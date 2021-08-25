import pandas as pd

def discretization(dataset, column, bins, labels=None, duplicates=None, mode=None):
    '''
    :param dataset: Pandas DataFrame
    :param column: specific column in the dataset
    :param bins: amount of bins to separate the values in the columns
    :param labels: rename the bins into specific name
    :param duplicates: remove duplicates bins
    :param mode: choose equal_width / equal frequency binning
    :return: discretization on column
    '''
    if duplicates == True:
        if mode == 'qcut':
            dataset[column] = pd.qcut(dataset[column], q=bins, labels=labels, duplicates='drop')
        else:
            dataset[column] = pd.cut(dataset[column], bins=bins, labels=labels)
    else:
        if mode == 'qcut':
            dataset[column] = pd.qcut(dataset[column], q=bins, labels=labels, duplicates='drop')
        else:
            dataset[column] = pd.cut(dataset[column], bins=bins, labels=labels)


def count_att(dataset, column, value):
    '''
    :param dataset: Pandas DataFrame
    :param column: specific column in the dataset
    :param value: which value in the column should be counted
    :return: probability of (value) to show in (column), included Laplacian correction
    '''
    try: # if (value) not been found then return laplacian calculation to preserve the probability
        p = dataset[column].value_counts()[value] / len(dataset)
        if (p == 0):
            p = 1 / (len(dataset) + len(dataset[column].value_counts()))
        return p
    except KeyError:
        return 1 / (len(dataset) + len(dataset[column].value_counts()))


def count_conditional_att(dataset, features_, f_label, class_, c_label):
    '''
    :param dataset: Pandas DataFrame
    :param features_: First selected column
    :param f_label: Second selected column
    :param class_: Independent value of column1
    :param c_label: Dependent value of column2
    :return: conditional probability of Y when X already occurred.
    P(A|B)=P(B|A)P(A)/P(B)
    P(class|features)=P(features|class) * P(class) / P(features)
    '''
    try: # if (f_label) not been found then return 1 to preserve the probability
        p = pd.crosstab(dataset[features_], dataset[class_], normalize='columns')[c_label][f_label]
        if (p == 0):
            p = 1
        return p
    except KeyError:
        return 1


def init():
    '''
    Initialize train and test data, drop all NaN values, make discretization on each of the Numeric columns
    :return: train and test DataFrames
    '''
    train = pd.read_csv('train.csv', delimiter=',')
    test = pd.read_csv('test.csv', delimiter=',')
    train.dropna(inplace=True)
    train.reset_index(drop=True, inplace=True)
    test.dropna(inplace=True)
    test.reset_index(drop=True, inplace=True)

    discretization(train, 'age', 5, [chr(i) for i in range(ord('A'), ord('E') + 1)], duplicates=True, mode='qcut')
    discretization(train, 'balance', 5, [chr(i) for i in range(ord('A'), ord('E') + 1)], duplicates=True, mode='qcut')
    discretization(train, 'duration', 5, [chr(i) for i in range(ord('A'), ord('E') + 1)], duplicates=True, mode='qcut')
    discretization(train, 'campaign', 5, [chr(i) for i in range(ord('A'), ord('C') + 1)], duplicates=True, mode='qcut')
    discretization(train, 'previous', 5, [chr(i) for i in range(ord('A'), ord('E') + 1)], mode='qut')

    discretization(test, 'age', 5, [chr(i) for i in range(ord('A'), ord('E') + 1)], duplicates=True, mode='qcut')
    discretization(test, 'balance', 5, [chr(i) for i in range(ord('A'), ord('E') + 1)], duplicates=True, mode='qcut')
    discretization(test, 'duration', 5, [chr(i) for i in range(ord('A'), ord('E') + 1)], duplicates=True, mode='qcut')
    discretization(test, 'campaign', 5, [chr(i) for i in range(ord('A'), ord('C') + 1)], duplicates=True, mode='qcut')
    discretization(test, 'previous', 5, [chr(i) for i in range(ord('A'), ord('E') + 1)], mode='qut')

    return train, test


def train(dataset):
    '''
    save all of the conditional probabilities in the dictionary for each of the values 'Age':['A': 0.2685]
    :param dataset: Pandas DataFrame
    :return: yes and no dictionaries that contain all of the conditional probabilities with 'class' [yes/no] respectively
    '''
    probability_yes = {}
    probability_no = {}
    probabilities = {'age': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                     'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student','blue-collar','self-employed', 'retired', 'technician', 'services'],
                     'marital': ['married', 'single', 'divorced'],
                     'education': ['tertiary', 'secondary', 'unknown', 'primary'],
                     'default': ['no', 'yes'],
                     'balance': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'],
                     'housing': ['yes', 'no'],
                     'loan': ['no', 'yes'],
                     'contact': ['unknown', 'cellular', 'telephone', 'Cellular'],
                     'month': ['may', 'jun', 'jul', 'JUL', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr','MAY', 'sep'],
                     'duration': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                     'campaign': ['A', 'B', 'C', 'D'],
                     'previous': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
                     'poutcome': ['unknown', 'failure', 'other', 'success', 'OTHER'],
                     'class': ['yes', 'no']
                     }

    # Calculate Probabilities of yes/no class
    for column, value in probabilities.items():
        for i in value:
            if (column != 'class'):
                probability_yes[(column, i)] = count_conditional_att(dataset, column, i, 'class', 'yes')
                probability_no[(column, i)] = count_conditional_att(dataset, column, i, 'class', 'no')

    return probability_yes, probability_no


def naive_bayes_classifier(dataset, yes_dict, no_dict, row):
    '''
    Get Test dataset, row in the test dataset to be classified, yes/no dictionaries
    :param dataset: Pandas DataFrame
    :param yes_dict: conditional probabilities that contain calculation with [class: 'yes']
    :param no_dict: conditional probabilities that contain calculation with [class: 'no']
    :param row: row of data to be classified
    :return: classified row with two options: yes/no
    '''
    prod_prob_y = 1
    prod_prob_n = 1
    for column in dataset:
        if (column != 'class'):
            try:
                prod_prob_y *= yes_dict[(column, dataset[column][row])]
            except KeyError:
                prod_prob_y *= 1
            try:
                prod_prob_n *= no_dict[(column, dataset[column][row])]
            except KeyError:
                prod_prob_n *= 1
    p_class_y = count_att(dataset, 'class', 'yes')
    p_class_n = count_att(dataset, 'class', 'no')
    if (prod_prob_y * p_class_y > prod_prob_n * p_class_n):
        return "yes"
    else:
        return "no"


def test(dataset, y_dict, n_dict):
    '''
    Calculate the success rate of (naive_bayes_classifier) and return the amount of correct predictions
    :param dataset: Pandas DataFrame
    :param y_dict: conditional probabilities that contain calculation with [class: 'yes']
    :param n_dict: conditional probabilities that contain calculation with [class: 'no']
    :return: amount of correct predictions
    '''
    count_correct = 0
    for row in range(len(dataset)):
        answer = naive_bayes_classifier(dataset=dataset, yes_dict=y_dict, no_dict=n_dict, row=row)
        real_answer = (dataset.iloc[row][-1])
        if (answer == real_answer):
            count_correct += 1
    return count_correct


# Load Information
train_data = init()[0]
test_data = init()[1]
yes_dict = train(train_data)[0]
no_dict = train(train_data)[1]
test_ = test(test_data, yes_dict, no_dict)
print("Total correct was: {0}/{1} | %{2}".format(test_, len(test_data),round((test_ / len(test_data)) * 100, ndigits=3)))
