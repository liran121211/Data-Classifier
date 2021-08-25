from os.path import basename
from pandas import read_csv
from NaiveBayes import *
from DecisionTrees import *
from KNN import *
from K_Means import *
from Evaluator import *

from PickleFiles import *


def run():
    try:
        os.mkdir(os.path.join("", "myFiles"))
    except FileExistsError:
        pass

    ask_to_load = input("Would you like to load model object?\n1) Yes\n2) No\nYour choice: ")

    if ask_to_load == '1':
        pickle_file = input("Enter dump file destination: ")
        file_dump = loadData(pickle_file)
        print("Start analysis...\n")
        analysis(file_dump)

    if ask_to_load == '2':
        discretization_mode = {'1': 'equal-width', '2': 'equal-frequency', '3': 'entropy'}
        train_path = input("Please enter training file location: ")
        test_path = input("Please enter testing file location: ")
        user_bins = int(input("\nEnter amount of bins: "))
        bin_mode = input("\nEnter discretization mode:\n1) Equal-Width\n2) Equal-Frequency\n3) Entropy\nYour choice: ")
        user_algorithm = input("\nEnter algorithm mode:\n"
                               "1) Decision Tree\n"
                               "2) SKLearn Decision Tree\n"
                               "3) Naive Bayes\n"
                               "4) SKLearn Naive Bayes\n"
                               "5) KNN\n"
                               "6) K-Means\n"
                               "Your choice: ")

        bin_mode = discretization_mode[bin_mode]
        train = read_csv(filepath_or_buffer=train_path, delimiter=',')
        test = read_csv(filepath_or_buffer=test_path, delimiter=',')

        if user_algorithm == '1':
            decision_tree = DecisionTree(train, test, basename(train_path), basename(test_path), 0.001, user_bins,
                                         bin_mode)
            decision_tree.run()
            analysis(decision_tree)
            if input("Would you like to save your object into (pk) file?\n1) Yes\n2) No\nYour choice: ") == '1':
                storeData(decision_tree)

        if user_algorithm == '2':
            decision_tree_sk = DecisionTreeSKLearn(train, test, 10, 10, basename(train_path), basename(test_path))
            decision_tree_sk.run()
            analysis(decision_tree_sk)
            if input("Would you like to save your object into (pk) file?\n1) Yes\n2) No\nYour choice: ") == '1':
                storeData(decision_tree_sk)

        if user_algorithm == '3':
            naive_bayes = NaiveBayes(train, test, basename(train_path), basename(test_path), user_bins, bin_mode)
            naive_bayes.run()
            analysis(naive_bayes)
            if input("Would you like to save your object into (pk) file?\n1) Yes\n2) No\nYour choice: ") == '1':
                storeData(naive_bayes)

        if user_algorithm == '4':
            naive_bayes_sk = NaiveBayes_SKLearn(train, test, basename(train_path), basename(test_path))
            naive_bayes_sk.run()
            analysis(naive_bayes_sk)
            if input("Would you like to save your object into (pk) file?\n1) Yes\n2) No\nYour choice: ") == '1':
                storeData(naive_bayes_sk)

        if user_algorithm == '5':
            knn = KNN(train, test, 5, basename(train_path), basename(test_path))
            knn.run()
            analysis(knn)
            if input("Would you like to save your object into (pk) file?\n1) Yes\n2) No\nYour choice: ") == '1':
                storeData(knn)

        if user_algorithm == '6':
            k_means = KMeans(train, 5, 100, 30)
            k_means.run()
            analysis(k_means)
            if input("Would you like to save your object into (pk) file?\n1) Yes\n2) No\nYour choice: ") == '1':
                storeData(k_means)


repeated = True
while (repeated):
    run()
    if input("\n\nRun Again?\n1) Yes\n2) No\nYour choice: ") == '2':
        repeated = False
