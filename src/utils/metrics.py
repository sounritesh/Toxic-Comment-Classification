from sklearn import metrics

def eval_perf(targets, outputs, threshold=0.5):
    y_pred_tag = [1 if i[0]>threshold else 0 for i in outputs]

    acc = metrics.accuracy_score(targets, y_pred_tag, normalize=True, sample_weight=None)
    prec, recall, fscore, _ = metrics.precision_recall_fscore_support(targets, y_pred_tag)
    roc_auc = metrics.roc_auc_score(targets, outputs)
    
    return acc, prec, recall, fscore, roc_auc