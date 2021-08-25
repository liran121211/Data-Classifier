import pandas as pd
from seaborn import heatmap
from matplotlib import pyplot as plt
from Preprocessing import categoricalToNumeric, validator
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
    if isinstance(y_test, pd.DataFrame):
        confusion_matrix = pd.crosstab(y_test.iloc[:, -1], y_prediction, rownames=['Actual'], colnames=['Predicted'])
    else:
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
    validator(confusion_matrix=confusion_matrix_data)
    TP = confusion_matrix_data.iloc[1, 1]
    TN = confusion_matrix_data.iloc[0, 0]
    TP_TN_FP_FN = confusion_matrix_data.values.sum()
    result = (TP + TN) / TP_TN_FP_FN

    if textual:
        print('\nAccuracy: %f' % result)

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
    validator(confusion_matrix=confusion_matrix_data)

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
    validator(confusion_matrix=confusion_matrix_data)
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
        print('F1-Score: %f' % result)

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
        print('SKLearn F1-Score: %f' % result, '\n')

    return result


def testAccuracy(obj):
    """
    Calculate the test accuracy of the given model
    :param obj: Model object
    :return: Test accuracy
    """
    if obj.__class__.__name__ == "NaiveBayes":
        print('Test Accuracy: %f' % (obj.score / len(obj.X_y_test)))
    elif obj.__class__.__name__ == "NaiveBayes_SKLearn":
        print('Test Accuracy: %f' % (obj.score / len(obj.y_test)))
    elif obj.__class__.__name__ == "DecisionTree":
        print('Test Accuracy: %f' % (obj.score / len(obj.test_data)))
    elif obj.__class__.__name__ == "DecisionTreeSKLearn":
        print('Test Accuracy: %f' % (obj.score / len(obj.y_test)))
    elif obj.__class__.__name__ == "KNN":
        print('Test Accuracy: %f' % (obj.score / len(obj.test_data)))
    elif obj.__class__.__name__ == "KMeans":
        print('Test Accuracy: %f' % (obj.score / len(obj.result)))
    else:
        print('Model [{0}] does not support testAccuracy() function'.format(obj.__class__.__name__))
        exit()


def trainAccuracy(obj):
    """
    Calculate the train accuracy of the given model
    :param obj: Model object
    :return: Test accuracy
    """
    if obj.__class__.__name__ == "NaiveBayes":
        obj.score = 0
        obj.y_prediction = []
        obj.X_y_test = obj.X_y_train
        obj.test(obj.class_probabilities[1], obj.class_probabilities[0])
        print('Train Accuracy: %f' % (obj.score / len(obj.X_y_test)))
    elif obj.__class__.__name__ == "NaiveBayes_SKLearn":
        obj.score = 0
        obj.X_test = obj.X_train
        obj.y_test = obj.y_train
        obj.y_prediction = []
        obj.run()
        print('Train Accuracy: %f' % (obj.score / len(obj.y_test)))
    elif obj.__class__.__name__ == "DecisionTree":
        obj.score = 0
        obj.y_prediction = []
        obj.test_data = obj.train_data
        obj.test()
        print('Train Accuracy: %f' % (obj.score / len(obj.test_data)))
    elif obj.__class__.__name__ == "DecisionTreeSKLearn":
        obj.score = 0
        obj.y_prediction = []
        obj.X_test = obj.X_train
        obj.y_test = obj.y_train
        obj.run()
        print('Train Accuracy: %f' % (obj.score / len(obj.y_test)))
    elif obj.__class__.__name__ == "KNN":
        obj.score = 0
        obj.y_prediction = []
        obj.test_data = obj.train_data
        obj.prediction()
        print('Train Accuracy: %f' % (obj.score / len(obj.test_data)))
    elif obj.__class__.__name__ == "KMeans":
        raise Exception('KMeans algorithm does not support TRAIN/TEST FILE')
    else:
        raise Exception('Model [{0}] does not support trainAccuracy() function'.format(obj.__class__.__name__))


def analysis(obj):
    """
    Run all evaluation tests on given object model
    :param obj: Model object
    :return: evaluation accuracies
    """
    print("Runing Analysis...\n")
    if obj.__class__.__name__ == "NaiveBayes":
        confusionMatrix(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, visual=True, textual=True)
        accuracy(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        accuracySKLearn(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        precision(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        precisionSKLearn(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        recall(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        recall_SKLearn(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        f1Score(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        f1ScoreSKLearn(y_test=obj.X_y_test.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        testAccuracy(obj=obj)
        trainAccuracy(obj=obj)
    elif obj.__class__.__name__ == "NaiveBayes_SKLearn":
        confusionMatrix(y_test=obj.y_test, y_prediction=obj.y_prediction, visual=True, textual=True)
        accuracy(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        accuracySKLearn(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        precision(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        precisionSKLearn(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        recall(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        recall_SKLearn(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        f1Score(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        f1ScoreSKLearn(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        testAccuracy(obj=obj)
        trainAccuracy(obj=obj)
    elif obj.__class__.__name__ == "DecisionTree":
        confusionMatrix(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, visual=True, textual=True)
        accuracy(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        accuracySKLearn(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        precision(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        precisionSKLearn(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        recall(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        recall_SKLearn(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        f1Score(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        f1ScoreSKLearn(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        testAccuracy(obj=obj)
        trainAccuracy(obj=obj)
    elif obj.__class__.__name__ == "DecisionTreeSKLearn":
        confusionMatrix(y_test=obj.y_test, y_prediction=obj.y_prediction, visual=True, textual=True)
        accuracy(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        accuracySKLearn(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        precision(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        precisionSKLearn(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        recall(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        recall_SKLearn(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        f1Score(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        f1ScoreSKLearn(y_test=obj.y_test, y_prediction=obj.y_prediction, textual=True)
        testAccuracy(obj=obj)
        trainAccuracy(obj=obj)
    elif obj.__class__.__name__ == "KNN":
        confusionMatrix(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, visual=True, textual=True)
        accuracy(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        accuracySKLearn(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        precision(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        precisionSKLearn(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        recall(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        recall_SKLearn(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        f1Score(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        f1ScoreSKLearn(y_test=obj.test_data.iloc[:, -1], y_prediction=obj.y_prediction, textual=True)
        testAccuracy(obj=obj)
        trainAccuracy(obj=obj)
    elif obj.__class__.__name__ == "KMeans":
        confusionMatrix(y_test=obj.class_, y_prediction=obj.result, visual=True, textual=True)
        accuracy(y_test=obj.class_, y_prediction=obj.result, textual=True)
        accuracySKLearn(y_test=obj.class_, y_prediction=obj.result, textual=True)
        precision(y_test=obj.class_, y_prediction=obj.result, textual=True)
        precisionSKLearn(y_test=obj.class_, y_prediction=obj.result, textual=True)
        recall(y_test=obj.class_, y_prediction=obj.result, textual=True)
        recall_SKLearn(y_test=obj.class_, y_prediction=obj.result, textual=True)
        f1Score(y_test=obj.class_, y_prediction=obj.result, textual=True)
        f1ScoreSKLearn(y_test=obj.class_, y_prediction=obj.result, textual=True)
        testAccuracy(obj=obj)
        trainAccuracy(obj=obj)
    else:
        print('Model [{0}] does not support testAccuracy() function'.format(obj.__class__.__name__))
        exit()
