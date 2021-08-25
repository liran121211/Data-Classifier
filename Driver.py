from Evaluator import *
from NaiveBayes import NaiveBayes_SKLearn

train = pd.read_csv('adult.csv', delimiter=',')
test = pd.read_csv('adult.csv', delimiter=',')
nb = NaiveBayes_SKLearn(train, test)
nb.run()

confusionMatrix(nb.y_test, nb.y_prediction, visual= True, textual= True)
accuracy(nb.y_test, nb.y_prediction, textual= True)
accuracySKLearn(nb.y_test, nb.y_prediction, textual= True)
precision(nb.y_test, nb.y_prediction, textual= True)
precisionSKLearn(nb.y_test, nb.y_prediction, textual= True)
recall(nb.y_test, nb.y_prediction, textual= True)
recall_SKLearn(nb.y_test, nb.y_prediction, textual= True)
f1Score(nb.y_test, nb.y_prediction, textual= True)
f1ScoreSKLearn(nb.y_test, nb.y_prediction, textual= True)






