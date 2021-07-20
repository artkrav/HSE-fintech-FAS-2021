import numpy as np
import sklearn.metrics as metrics
from collections import OrderedDict
from typing import Dict
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def get_eer(targets, scores):
    fpr, tpr, thresholds = metrics.roc_curve(targets, scores, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer).item()
    return eer, thresh


def get_hter(targets, scores, threshold):
    far, tpr, thresholds = metrics.roc_curve(targets, scores, pos_label=1)
    frr = 1-tpr
    ind = np.abs(thresholds - threshold).argmin(axis=0)
    far_at, frr_at, thr_at = far[ind], frr[ind], thresholds[ind]
    hter = (far_at + frr_at) / 2
    return hter, thr_at


class AverageMeter:
    def __init__(self):
        self.accum = 0.
        self.n = 0.
        
    def append(self, value):
        self.accum += float(value)
        self.n += 1
        
    def reset(self):
        self.accum = 0.
        self.n = 0.
        
    @property
    def value(self):
        if self.n > 0:
            return self.accum / self.n
        return 0


class Tracker:
    def __init__(self):
        self._targets = []
        self._scores = []
    
    def reset(self):
        self._targets = []
        self._scores = []
        
    @property
    def targets(self):
        if isinstance(self._targets, list):
            self._targets = np.array(self._targets, dtype='int32')
        return self._targets
    
    @property
    def scores(self):
        if isinstance(self._scores, list):
            self._scores = np.array(self._scores, dtype='float32')
        return self._scores
    
    def append(self, score, target):
        if isinstance(self._scores, np.ndarray):
            self._scores = self._scores.tolist()
        if isinstance(self._targets, np.ndarray):
            self._targets = self._targets.tolist()
        self._scores.append(score)
        self._targets.append(target)
        
    def append_batch(self, scores, targets):
        scores = scores.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        for i in range(scores.shape[0]):
            self.append(score=scores[i].item(),
                        target=targets[i].item())
            
    def score(self, score, **kwargs):
        assert score in {'auprc', 'auroc', 'eer', 'hter'}
        
        if score == 'auprc':
            precision, recall, thresholds = metrics.precision_recall_curve(self.targets, self.scores)
            auprc = metrics.auc(x=recall, y=precision)
            return auprc
        elif score == 'auroc':
            return metrics.roc_auc_score(self.targets, self.scores)
        elif score == 'eer':
            return get_eer(self.targets, self.scores)
        elif score == 'hter':
            return get_hter(self.targets, self.scores, **kwargs)
        
    def auprc(self):
        return self.score('auprc')
    
    def auroc(self):
        return self.score('auroc')
    
    def eer(self):
        return self.score('eer')
    
    def hter(self, thr):
        return self.score('hter', threshold=thr)
