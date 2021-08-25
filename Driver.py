import pandas as pd

from NaiveBayes import *
from DecisionTrees import *
from KNN import *
from K_Means import *

ERROR_LOOP = True
while (ERROR_LOOP):
    try:
        # Initialize process
        train_path = input("\nEnter train file path: ")
        test_path = input("\nEnter train file path: ")
        user_bins = int(input("\nEnter amount of bins: "))
        user_discretization_mode = input("\nEnter Discretization Mode [equal-width/equal-frequency/entropy]: ")
        train = pd.read_csv(train_path, delimiter=',')[:100]
        test = pd.read_csv(test_path, delimiter=',')[:100]

        decision_tree_sk = DecisionTreeSKLearn(train_data=train, test_data=test, max_depth=10, random_state=10)
        decision_tree = DecisionTree(train_data=train, test_data=test, threshold=0.001, bins=user_bins,discretization_mode=user_discretization_mode)
        naive_bayes = NaiveBayes(train_file=train, test_file=test, bins=user_bins,discretization_mode=user_discretization_mode)
        naive_bayes_sk = NaiveBayes_SKLearn(train_file=train, test_file=test)
        knn = KNN(train_file=train, test_file=test, k_neighbors=5)
        k_means = KMeans(train_data=train, k_means=5, max_iterations=100, random_state=30)
        ERROR_LOOP = False

        # Run process
        print('\nDecision Tree Started:')
        decision_tree.run()
        print('\nSKLearn Decision Tree Started:')
        decision_tree_sk.run()
        print('\nNaive Bayes Started:')
        naive_bayes.run()
        print('\nSKLearn Naive Bayes Started:')
        naive_bayes_sk.run()
        print('\nKNN Started:')
        knn.run()
        print('\nK-Means Started:')
        k_means.run()



    except IndexError:
        print('Input is invalid, try again...')
        print(ValueError)
        ERROR_LOOP = True
    except FileNotFoundError:
        print('File does not exist, try again...')
        print(FileNotFoundError)
        ERROR_LOOP = True

