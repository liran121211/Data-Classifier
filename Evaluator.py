#Todo: PRECISION / RECALL/ F-MEASURE / ACCUARACY / CONFUSION MATRIX

"""
precision
tp / (tp+fp)
true positive and false positives
precision is calculated as the number of true positives divided
by the total number of true positives and false positives.

"""

# calculates precision for 1:100 dataset with 90 tp and 30 fp
from sklearn.metrics import precision_score
# define actual
pos = [1 for i in range(100)]
neg = [0 for i in range(10000)]
y_true = pos + neg
# define predictions
prediction_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
prediction_neg = [1 for _ in range(30)] + [0 for _ in range(9970)]
y_pred = prediction_pos + pred_neg
# calculate prediction
precision = precision_score(y_true, y_pred, average='binary')
print('Precision: %.3f' % precision)



"""
recall
tp / (tp +fn)
true positives and false positives 
"""

# calculates recall for 1:100 dataset with 90 tp and 10 fn
from sklearn.metrics import recall_score
# define actual
act_pos = [1 for _ in range(100)]
act_neg = [0 for _ in range(10000)]
y_true = act_pos + act_neg
# define predictions
pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
pred_neg = [0 for _ in range(10000)]
y_pred = pred_pos + pred_neg
# calculate recall
recall = recall_score(y_true, y_pred, average='binary')
print('Recall: %.3f' % recall)


