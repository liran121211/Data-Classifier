import pandas as pd
from seaborn import heatmap
from matplotlib import pyplot as plt
from Preprocessing import categoricalToNumeric
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def confusionMatrix(y_test, y_prediction, visual=False, textual=False):
    """
    Create error matrix that allows visualization of the performance of prediction model
    True Positive (TP): True positive represents the value of correct predictions of positives out of positive cases.
    False Positive (FP): False positive represents the value of incorrect positive predictions.
    True Negative (TN): True negative represents the value of correct predictions of negatives out of negative cases.
    False Negative (FN): False negative represents the value of incorrect negative predictions.
    ╔════╦════╗
    ║ TN ║ FP ║
    ╠════╬════╣
    ║ FN ║ TP ║
    ╚════╩════╝
    :param y_test: classification column of test dataset
    :param y_prediction: classification column of prediction array
    :param visual: true to show visual matrix else false
    :param textual:  true to print matrix else false
    :return: Confusion Matrix
    """
    # Transfer data to numeric
    y_test = categoricalToNumeric(y_test)
    y_prediction = categoricalToNumeric(y_prediction)

    # Calculate Confusion Matrix
    confusion_matrix = pd.crosstab(y_test, y_prediction, rownames=['Actual'], colnames=['Predicted'])
    if textual:
        print(confusion_matrix)

    # Visual Matrix
    if visual:
        heatmap(confusion_matrix, annot=True, fmt='d')
        plt.show()

    return confusion_matrix


def accuracy(y_test, y_prediction, textual=False):
    """
    Represents the model’s ability to correctly predict both the positives and negatives out of all the predictions.
    #(TP + TN) / ([TP + FP] + [FN + TN])
    :param y_test: classification vector of test data
    :param y_prediction: classification vector of prediction data
    :param textual:  true to print matrix else false
    :return: Accuracy percentage
    """
    confusion_matrix_data = confusionMatrix(y_test, y_prediction)
    TP = confusion_matrix_data.iloc[1, 1]
    TN = confusion_matrix_data.iloc[0, 0]
    TP_TN_FP_FN = confusion_matrix_data.values.sum()
    result = (TP + TN) / TP_TN_FP_FN

    if textual:
        print('Accuracy: %f' % result)

    return result


def accuracySKLearn(y_test, y_prediction, textual=False):
    """
    Represents the model’s ability to correctly predict both the positives and negatives out of all the predictions.
    #(TP + TN) / ([TP + FP] + [FN + TN])
    :param y_test: classification vector of test data
    :param y_prediction: classification vector of prediction data
    :param textual:  true to print matrix else false
    :return: Accuracy percentage
    """
    y_test = categoricalToNumeric(y_test)
    y_prediction = categoricalToNumeric(y_prediction)
    result = accuracy_score(y_test, y_prediction)

    if textual:
        print('SKLearn Accuracy: %f' % result)

    return result


def precision(y_test, y_prediction, textual=False):
    """
    Represents the model’s ability to correctly predict the positives out of all the positive prediction it made.
    Precision score is a useful measure of success of prediction when the classes are very imbalanced.
    #TP / (TP + FP)
    :param y_test: classification vector of test data
    :param y_prediction: classification vector of prediction data
    :param textual:  true to print matrix else false
    :return: Precision percentage
    """
    confusion_matrix_data = confusionMatrix(y_test, y_prediction)
    TP = confusion_matrix_data.iloc[1, 1]
    FP = confusion_matrix_data.iloc[0, 1]
    result = TP / (TP + FP)

    if textual:
        print('Precision: %f' % result)

    return result


def precisionSKLearn(y_test, y_prediction, textual=False):
    """
    Represents the model’s ability to correctly predict the positives out of all the positive prediction it made.
    Precision score is a useful measure of success of prediction when the classes are very imbalanced.
    #TP / (TP + FP)
    :param y_test: classification vector of test data
    :param y_prediction: classification vector of prediction data
    :param textual:  true to print matrix else false
    :return: Precision percentage
    """
    y_test = categoricalToNumeric(y_test)
    y_prediction = categoricalToNumeric(y_prediction)
    result = precision_score(y_test, y_prediction)

    if textual:
        print('SKLearn Precision: %f' % result)

    return result


def recall(y_test, y_prediction, textual=False):
    """
    Represents the model’s ability to correctly predict the positives out of actual positives.
    This is unlike precision which measures as to how many predictions made by models are actually positive out of all
    ,positive predictions made
    :param y_test: classification vector of test data
    :param y_prediction: classification vector of prediction data
    :param textual:  true to print matrix else false
    :return: Recall percentage
    """
    confusion_matrix_data = confusionMatrix(y_test, y_prediction)
    TP = confusion_matrix_data.iloc[1, 1]
    FN = confusion_matrix_data.iloc[1, 0]
    result = (TP / (TP + FN))

    if textual:
        print('Recall: %f' % result)

    return result


def recall_SKLearn(y_test, y_prediction, textual=False):
    """
    Represents the model’s ability to correctly predict the positives out of actual positives.
    This is unlike precision which measures as to how many predictions made by models are actually positive out of all
    ,positive predictions made
    :param y_test: classification vector of test data
    :param y_prediction: classification vector of prediction data
    :param textual:  true to print matrix else false
    :return: Recall percentage
    """
    y_test = categoricalToNumeric(y_test)
    y_prediction = categoricalToNumeric(y_prediction)
    result = recall_score(y_test, y_prediction)

    if textual:
        print('SKLearn Recall: %f' % result)

    return result


def f1Score(y_test, y_prediction, textual=False):
    """
    Represents the model score as a function of precision and recall score
    This is useful measure of the model in the scenarios where one tries to optimize either of precision or recall score
    ,and as a result, the model performance suffers
    :param y_test: classification vector of test data
    :param y_prediction: classification vector of prediction data
    :param textual:  true to print matrix else false
    :return: Recall percentage
    """
    precision_score_ = precision(y_test, y_prediction)
    recall_Score = recall(y_test, y_prediction)
    result = (2 * precision_score_ * recall_Score) / (precision_score_ + recall_Score)

    if textual:
        print('F1 score: %f' % result)

    return result


def f1ScoreSKLearn(y_test, y_prediction, textual=False):
    """
    Represents the model score as a function of precision and recall score
    This is useful measure of the model in the scenarios where one tries to optimize either of precision or recall score
    ,and as a result, the model performance suffers
    :param y_test: classification vector of test data
    :param y_prediction: classification vector of prediction data
    :param textual:  true to print matrix else false
    :return: Recall percentage
    """
    y_test = categoricalToNumeric(y_test)
    y_prediction = categoricalToNumeric(y_prediction)
    result = f1_score(y_test, y_prediction)

    if textual:
        print('SKLearn Recall: %f' % result)

    return result

def trainAccuracy():

