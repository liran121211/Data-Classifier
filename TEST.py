import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from NaiveBayes import NaiveBayes_SKLearn, NaiveBayes
from Preprocessing import categoricalToNumeric
from DecisionTrees import DecisionTree, DecisionTreeSKLearn

train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')
nb = NaiveBayes(train, test)
nb.run()


# x = categoricalToNumeric(nb.y_test)
# y = categoricalToNumeric(nb.y_prediction)
#
# confusion_matrix = pd.crosstab(x, y, rownames=['Actual'], colnames=['Predicted'])
#
# sn.heatmap(confusion_matrix, annot=True)
# plt.show()
