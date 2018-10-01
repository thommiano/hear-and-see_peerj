def f1_score(confusion_matrix,class_list):
    '''F1-Score is the harmonic mean of precision and recall.
    f1 = 2 * [ (precision * recall) / (precision + recall) ]
    '''
    f1_score = {}
    for label in class_list:
        numerator = 2. * confusion_matrix[label, label]
        denominator = (np.sum(confusion_matrix[label, :]) + np.sum(confusion_matrix[:, label]))
        f1_score[label] = numerator/denominator
    return f1_score