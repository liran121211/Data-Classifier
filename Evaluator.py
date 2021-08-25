# Todo: PRECISION / RECALL/ F-MEASURE / ACCUARACY / CONFUSION MATRIX

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

from Preprocessing import categoricalToNumeric

"""
Precision = TruePositives / (TruePositives + FalsePositives)
true positive and false positives
precision is calculated as the number of true positives divided
by the total number of true positives and false positives.

"""


# calculates precision for 1:100 dataset with 90 tp and 30 fp
def precision(y_true):
    # define predictions
    # predicting 90 true positives
    prediction_pos = [0 for i in range(20)] + [1 for j in range(80)]
    # predicting 30 true positives
    prediction_neg = [1 for i in range(0)] + [0 for j in range(10000)]
    # sum predictions by true positive from prediction_pos / (prediction_pos + prediction_neg (falsePositive))
    y_prediction = prediction_pos + prediction_neg
    # calculate prediction
    precision = precision_score(y_true, y_prediction, average='binary')
    print('Precision: %.3f' % precision)


"""
Recall = TruePositives / (TruePositives + FalseNegatives)
recall is calculated as the number of true positives divided
by the total number of true positives and false negatives.
"""


def recall(y_true):
    # define predictions
    # false negative + true positive
    prediction_pos = [0 for i in range(20)] + [1 for j in range(80)]
    # false positive
    prediction_neg = [0 for i in range(10000)]
    y_prediction = prediction_pos + prediction_neg
    # calculate recall
    recall = recall_score(y_true, y_prediction, average='binary')
    print('Recall: %.3f' % recall)


"""
F-Measure = (2 * Precision * Recall) / (Precision + Recall)

"""


def fMeasure(y_true):
    # define predictions
    pred_pos = [0 for i in range(5)] + [1 for j in range(95)]
    pred_neg = [1 for i in range(55)] + [0 for j in range(9945)]
    y_pred = pred_pos + pred_neg
    # calculate score
    score = f1_score(y_true, y_pred, average='binary')
    print('F-Measure: %.3f' % score)


# Cross Validation Classification Accuracy
def Accuaracy(y_true, y_prediction):
    
    
    accuracy = accuracy_score(prediction, labels_test)

    print("Accuracy: %.3f (%.3f)" % (results.mean() * 100, results.std() * 100))


def confusionMatrix(y_true, y_prediction):
    """
    Create error matrix that allows visualization of the performance of prediction model
    :param y_true: classification column of test dataset
    :param y_prediction: classification column of prediction array
    :return: Confusion Matrix
    """
    if (isinstance(y_prediction, str)):
        y_prediction = categoricalToNumeric(y_prediction)
    return confusion_matrix(y_true, y_prediction)


# # define actual positives and negatives
# actual_pos = [1 for i in range(100)]  # true positives
# actual_neg = [0 for i in range(10000)]  # true negatives
# y_true = actual_pos + actual_neg
# precision(y_true)
# recall(y_true)
# fMeasure(y_true)
# Accuaracy()

