from sklearn import metrics

def eval_perf(targets, outputs, threshold=0.5):
    '''
    Function to calculate result metrics

    Parameters
    targets (torch.Tensor): expected labels.
    outputs (torch.Tensor): output logits from the model.
    threshold (float): threshold for logits' classification.

    Returns
    acc (float): Accuracy of the predictions
    prec (List(float)): List containing precision of both classes.
    recall (List(float)): List containing recall of both classes.
    fscore (List(float)): List containing f1 score of both classes.
    roc_auc (float): Area Under Curve ROC of the predictions.
    '''
    y_pred_tag = [1 if i[0]>threshold else 0 for i in outputs]

    acc = metrics.accuracy_score(targets, y_pred_tag, normalize=True, sample_weight=None)
    prec, recall, fscore, _ = metrics.precision_recall_fscore_support(targets, y_pred_tag, average='macro')
    roc_auc = metrics.roc_auc_score(targets, outputs)
    pr_auc = metrics.average_precision_score(targets, outputs)

    return acc, prec, recall, fscore, roc_auc, pr_auc