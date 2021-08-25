import K_Means
from Evaluator import *
from NaiveBayes import *
from DecisionTrees import *
from KNN import *
from K_Means import *

train = pd.read_csv('german_credit.csv', delimiter=',')[:5]
test = pd.read_csv('german_credit.csv', delimiter=',')
decision_tree_sk = DecisionTreeSKLearn(train_data=train, test_data=test, max_depth=10)
decision_tree = DecisionTree(train_data=train, test_data=test, threshold=0.001)
naive_bayes = NaiveBayes(train_file=train, test_file=test)
naive_bayes_sk = NaiveBayes_SKLearn(train_file=train, test_file=test)
knn = KNN(train_file=train, test_file=test, k_neighbors=5)
k_means = KMeans(train_data=train, k_means=5, max_iterations=100, random_state=30)


trainAccuracy(FloatingPointError)

# testAccuracy(nb)
# confusionMatrix(nb.y_test, nb.y_prediction, visual=True, textual=True)
# accuracy(nb.y_test, nb.y_prediction, textual=True)
# accuracySKLearn(nb.y_test, nb.y_prediction, textual=True)
# precision(nb.y_test, nb.y_prediction, textual=True)
# precisionSKLearn(nb.y_test, nb.y_prediction, textual=True)
# recall(nb.y_test, nb.y_prediction, textual=True)
# recall_SKLearn(nb.y_test, nb.y_prediction, textual=True)
# f1Score(nb.y_test, nb.y_prediction, textual=True)
# f1ScoreSKLearn(nb.y_test, nb.y_prediction, textual=True)
