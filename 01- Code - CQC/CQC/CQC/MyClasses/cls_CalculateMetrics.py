from operator import truediv
import numpy as np


class cls_CalculateMetrics():


    def __init__(self, cm):
        self.cm = cm # it is a numpy object
        self.true_positives = np.diag(cm)
    
    
    # calculate precision for each class
    def calc_precision(self):
        columns_sum = np.sum(self.cm, axis=0)
        prec = list(map(truediv, self.true_positives, columns_sum))
        self.precision = prec
        return prec
    
    
    # calculate recall for each class
    # recall = accuracy per class
    # how accuratly each class is predicted
    def calc_recall(self):
        rows_sum = np.sum(self.cm, axis=1)
        rec = list(map(truediv, self.true_positives, rows_sum))
        self.recall = rec
        return rec
    
    
    # calculate f1_score for each class
    def calc_f1_score(self):
        tempPrec = np.array(self.precision)
        tempRec = np.array(self.recall)
        numerator = tempPrec * tempRec
        Denominator = tempPrec + tempRec
        f1s = 2 * (numerator / Denominator)
        self.f1_score = f1s
        return f1s
    

    def calc_accuracy(self):
        total_samples = np.sum(self.cm)
        sum_true_positives = sum(self.true_positives)
        acc = (sum_true_positives / total_samples)
        return acc



