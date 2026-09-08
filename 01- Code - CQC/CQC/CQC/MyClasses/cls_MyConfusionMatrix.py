import numpy as np


# MyConfusionMatrix
class cls_MyConfusionMatrix():


    def __init__(self, num_classes):
        # rows: actual, columns: prediction
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    
    def update(self, y, yhat_indices):
        for actual, pred in zip(y, yhat_indices):
            self.confusion_matrix[actual, pred] += 1
    
    
    def calc_accuracy(self):
        diagon = self.confusion_matrix.diagonal()
        # accuracy
        total_samples = self.confusion_matrix.sum()
        total_corrects = diagon.sum()
        accuracy = 100 * (total_corrects / total_samples)
        
        # accuracy per class
        # sum(1): 1 referes to sum for each row
        samples_per_class = self.confusion_matrix.sum(1)
        accuracy_per_class = 100 * (np.divide(diagon, samples_per_class))
        
        return accuracy, accuracy_per_class.tolist()
    
    
    def get_cf(self):
        return self.confusion_matrix.tolist()




