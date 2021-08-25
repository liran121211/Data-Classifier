import pandas as pd

from NaiveBayes import *
from DecisionTrees import *
from KNN import *
from K_Means import *

ERROR_LOOP = True
DISCRETIZATION_MODE = {'1': 'equal-width', '2': 'equal-frequency', '3': 'entropy'}

while (ERROR_LOOP):
    try:
        # Initialize process
        train_path = "C:\\Users\\Liran\\PycharmProjects\\Final_project\\historical-senate-predictions.csv"
        test_path = "historical-senate-predictions.csv"
        user_bins = int(input("\nEnter amount of bins: "))
        bin_mode = input("\nEnter discretization mode:\n1) Equal-Width\n2) Equal-Frequency\n3) Entropy\nYour choice: ")
        bin_mode = DISCRETIZATION_MODE[bin_mode]
        train = pd.read_csv(train_path, delimiter=',')
        test = pd.read_csv(test_path, delimiter=',')

        decision_tree_sk = DecisionTreeSKLearn(train_data=train, test_data=test, max_depth=10, random_state=10,
                                               train_file_name=os.path.basename(train_path),
                                               test_file_name=os.path.basename(test_path))
        decision_tree = DecisionTree(train_data=train, test_data=test, train_file_name=os.path.basename(train_path),
                                     test_file_name=os.path.basename(test_path), threshold=0.001, bins=user_bins,
                                     discretization_mode=bin_mode)
        naive_bayes = NaiveBayes(train_file=train, test_file=test, bins=user_bins, discretization_mode=bin_mode)
        naive_bayes_sk = NaiveBayes_SKLearn(train_file=train, test_file=test)
        knn = KNN(train_file=train, test_file=test, k_neighbors=5)
        k_means = KMeans(train_data=train, k_means=5, max_iterations=100, random_state=30)
        ERROR_LOOP = False

        # Run process
        print('\nDecision Tree Started:')
        decision_tree.run()
        print('-' * 50)
        print('\nSKLearn Decision Tree Started:')
        decision_tree_sk.run()
        print('-' * 50)
        print('\nNaive Bayes Started:')
        naive_bayes.run()
        print('-' * 50)
        print('\nSKLearn Naive Bayes Started:')
        naive_bayes_sk.run()
        print('-' * 50)
        print('\nKNN Started:')
        knn.run()
        print('-' * 50)
        print('\nK-Means Started:')
        k_means.run()
        print('-' * 50)



    except ValueError:
        print('Input is invalid, try again...')
        print(ValueError)
        ERROR_LOOP = True
    except FileNotFoundError:
        print('File does not exist, try again...')
        print(FileNotFoundError)
        ERROR_LOOP = True

# train = pd.read_csv("historical-senate-predictions.csv", delimiter=',')[:50]
# test = pd.read_csv("historical-senate-predictions.csv", delimiter=',')[51:]
# decision_tree_sk = DecisionTreeSKLearn(train_data=train, test_data=test, max_depth=10, random_state=10)
# decision_tree_sk.run()
