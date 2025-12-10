import os, sys
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from scipy import stats


__all__ = [
    "pearsonr",
    "rsquare",
    "accuracy",
    "roc",
    "pr",
    "calculate_metrics"
]


# class MLMetrics(object):
class MLMetrics(object):
    """
    Class to track and aggregate multi-label or binary classification metrics.
    Stores metrics for each batch, computes cumulative sum and average,
    and provides access to key metrics like accuracy, ROC-AUC, PR-AUC,
    TP, TN, FP, FN, F1, precision, recall, MCC, and specificity.
    Can also store additional custom metrics passed via other_lst.
    """

    def __init__(self, objective='binary'):
        self.objective = objective
        self.metrics = []

    def update(self, label, pred, other_lst):
        met, _ = calculate_metrics(label, pred, self.objective)
        if len(other_lst) > 0:
            met.extend(other_lst)
        self.metrics.append(met)
        self.compute_avg()

    def compute_avg(self):
        if len(self.metrics) > 1:
            self.avg = np.array(self.metrics).mean(axis=0)
            self.sum = np.array(self.metrics).sum(axis=0)
        else:
            self.avg = self.metrics[0]
            self.sum = self.metrics[0]
        self.acc = self.avg[0]
        self.auc = self.avg[1]
        self.prc = self.avg[2]
        self.tp = int(self.sum[3])
        self.tn = int(self.sum[4])
        self.fp = int(self.sum[5])
        self.fn = int(self.sum[6])
        self.f1_score = self.avg[7]
        self.precision = self.avg[8]
        self.recall = self.avg[9]
        self.mcc = self.avg[10]
        self.specificity = self.avg[11]
        if len(self.avg) > 11:
            self.other = self.avg[11:]


def pearsonr(label, prediction):
    """
    Compute :contentReference[oaicite:0]{index=0}.
    If 1D, returns a single correlation;
    if 2D, returns correlation values for each column.
    """
    ndim = np.ndim(label)
    if ndim == 1:
        corr = [stats.pearsonr(label, prediction)]
    else:
        num_labels = label.shape[1]
        corr = []
        for i in range(num_labels):
            # corr.append(np.corrcoef(label[:,i], prediction[:,i]))
            corr.append(stats.pearsonr(label[:, i], prediction[:, i])[0])

    return corr


def rsquare(label, prediction):
    """
    Compute :contentReference[oaicite:1]{index=1} and slope.
    If 1D, returns single R² and slope;
    if 2D, returns values for each column.
    """
    ndim = np.ndim(label)
    if ndim == 1:
        y = label
        X = prediction
        m = np.dot(X, y) / np.dot(X, X)
        resid = y - m * X;
        ym = y - np.mean(y);
        rsqr2 = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym);
        metric = [rsqr2]
        slope = [m]
    else:
        num_labels = label.shape[1]
        metric = []
        slope = []
        for i in range(num_labels):
            y = label[:, i]
            X = prediction[:, i]
            m = np.dot(X, y) / np.dot(X, X)
            resid = y - m * X;
            ym = y - np.mean(y);
            rsqr2 = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym);
            metric.append(rsqr2)
            slope.append(m)
    return metric, slope


def accuracy(label, prediction):
    """
    Compute :contentReference[oaicite:2]{index=2} score.
    If 1D, returns a single value;
    if 2D, returns accuracy for each column.
    """

    ndim = np.ndim(label)
    if ndim == 1:
        metric = np.array(accuracy_score(label, np.round(prediction)))
    else:
        num_labels = label.shape[1]
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            metric[i] = accuracy_score(label[:, i], np.round(prediction[:, i]))
    return metric


def roc(label, prediction):
    """
    Compute :contentReference[oaicite:3]{index=3} (ROC)
    and :contentReference[oaicite:4]{index=4}.
    If 1D, returns single AUC and (fpr,tpr);
    if 2D, returns AUCs and curves for each column.
    """

    ndim = np.ndim(label)
    if ndim == 1:
        fpr, tpr, thresholds = roc_curve(label, prediction)
        score = auc(fpr, tpr)
        metric = np.array(score)
        curves = [(fpr, tpr)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            fpr, tpr, thresholds = roc_curve(label[:, i], prediction[:, i])
            score = auc(fpr, tpr)
            metric[i] = score
            curves.append((fpr, tpr))
    return metric, curves

def pr(label, prediction):
    """
    Compute :contentReference[oaicite:0]{index=0}
    and :contentReference[oaicite:1]{index=1}.
    If 1D, returns a single AUC and (precision, recall);
    if 2D, returns AUCs and curves for each column.
    """

    ndim = np.ndim(label)
    if ndim == 1:
        precision, recall, thresholds = precision_recall_curve(label, prediction)
        score = auc(recall, precision)
        metric = np.array(score)
        curves = [(precision, recall)]
    else:
        num_labels = label.shape[1]
        curves = []
        metric = np.zeros((num_labels))
        for i in range(num_labels):
            precision, recall, thresholds = precision_recall_curve(label[:, i], prediction[:, i])
            score = auc(recall, precision)
            metric[i] = score
            curves.append((precision, recall))
    return metric, curves


def tfnp(label, prediction):
    """
    Compute :contentReference[oaicite:0]{index=0} values:
    True Positives, True Negatives, False Positives, False Negatives.
    If computation fails, return all zeros.
    """
    try:
        tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
    except Exception:
        tp, tn, fp, fn = 0, 0, 0, 0

    return tp, tn, fp, fn


def calculate_metrics(label, prediction, objective):
    """
    Calculate classification metrics for binary or hinge objectives.
    Computes accuracy, ROC-AUC, PR-AUC, confusion matrix (TP, TN, FP, FN),
    precision, recall, F1 score, specificity, and Matthews correlation coefficient (MCC).
    Returns mean and standard deviation of key metrics.
    """

    if (objective == "binary") | (objective == 'hinge'):
        ndim = np.ndim(label)
        # if ndim == 1:
        #    label = one_hot_labels(label)
        correct = accuracy(label, prediction)
        auc_roc, roc_curves = roc(label, prediction)
        auc_pr, pr_curves = pr(label, prediction)
        # import pdb; pdb.set_trace()
        if ndim == 2:
            prediction = prediction[:, 0]
            label = label[:, 0]
        # pred_class = prediction[:,0]>0.5
        pred_class = prediction > 0.5
        # tp, tn, fp, fn = tfnp(label[:,0], pred_class)
        tp, tn, fp, fn = tfnp(label, pred_class)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / denominator if denominator > 0 else 0

        # tn8, fp8, fn8, tp8 = tfnp(label[:,0], prediction[prediction>0.8][:,0])
        # import pdb; pdb.set_trace()
        mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr), tp, tn, fp, fn, np.nanmean(f1_score),
                np.nanmean(precision), np.nanmean(recall), np.nanmean(mcc), np.nanmean(specificity)]
        std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr), np.nanstd(f1_score)]

    else:
        mean = 0
        std = 0

    return [mean, std]