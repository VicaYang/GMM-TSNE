import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from IncrementalTSNE import IncrementalTSNE
import copy

class TSNE(object):
    def __init__(self, X, gmm):
        self.X = X
        self.gmm = gmm
        self.result = None
        self.label_result = None

    def sample(self):
        idx = np.random.choice(len(self.X), len(self.X) // 3, replace=False)
        return self.X[idx], self.result[idx], self.gmm['classes'][idx].astype(int)


    def layout(self):
        alpha = 0.5
        beta = 0.5
        if self.result is not None:
            constraint_X, constraint_Y, constraint_classes = self.sample()
            classes = self.gmm['classes']
            n_labels = len(self.label_result[2])
            constraint_weight = np.ones((len(constraint_X), ), dtype=float).astype(dtype=np.float64) 
            [con_X, con_Y, con_w] = self.label_result
            if len(con_X) > 1:
                con_X = np.array(con_X)
                con_Y = np.array(con_Y)
                con_w = np.array(con_w)
                constraint_X = np.concatenate((constraint_X, con_X))
                constraint_Y = np.concatenate((constraint_Y, con_Y))
                constraint_weight = np.concatenate((constraint_weight, con_w))
                constraint_classes = np.concatenate((constraint_classes, np.array([i for i in range(n_labels)])))
            tsne = IncrementalTSNE(n_components=2, init=self.result, perplexity=30, angle=0.3, n_jobs=8, n_iter=500, random_state=42)
            self.result, self.label_result = tsne.fit_transform(self.X,
                                            constraint_X=constraint_X, constraint_Y=constraint_Y,
                                            gmm=self.gmm, label_alpha=alpha, label_beta=beta,
                                            labels=classes, constraint_labels=constraint_classes, constraint_weight=constraint_weight)
        else:
            classes = self.gmm['classes']
            tsne = IncrementalTSNE(n_components=2, init='pca', perplexity=30, angle=0.3, n_jobs=8, n_iter=1000, random_state=42)
            self.result, self.label_result = tsne.fit_transform(self.X, labels=classes, gmm=self.gmm, label_alpha=alpha, label_beta=beta)

        return {
            'x': self.result[:, 0],
            'y': self.result[:, 1],
        }