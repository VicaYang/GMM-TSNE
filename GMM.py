import numpy as np
from sklearn import mixture

gmm_max_iter = 10000


class GMM(object):

    def __init__(self, X):
        self.X = X
        self.gmm_result = GMM.build_gmm(self.X)

    @staticmethod
    def build_gmm(X, n_components=10):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        gmm = GMM._build_gmm(X, 'BIC', n_components=n_components)
        predict = gmm.predict(X)

        labels = np.unique(predict)
        sinvs = {label: np.linalg.pinv(gmm.covariances_[label]) for label in labels}
        log_dets = {label: np.linalg.slogdet(gmm.covariances_[label])[1] for label in labels}

        return {
            'classes': gmm.predict(X),
            'distribution': {
                i: {
                    'mu': gmm.means_[label],
                    's': gmm.covariances_[label],
                    'log_det': log_dets[label],
                    'sinv': sinvs[label],
                    'sp': gmm.weights_[label] * len(X),
                    'n': len(X),
                    'weight': gmm.weights_[label],
                    'log_weight': np.log(gmm.weights_[label]),
                } for i, label in enumerate(labels)
            },
            'labels': labels,
        }

    @staticmethod
    def _build_gmm(X, method, n_components=10):
        if method not in ['bayesian', 'BIC']:
            raise NotImplementedError(f"Please implement the method {method}")

        if method == 'bayesian':
            gmm = mixture.BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=1e-2,
                                                    mean_precision_prior=1e-2, max_iter=gmm_max_iter, n_init=5)
            gmm.fit(X)
            return gmm

        if method == 'BIC':
            models = []
            max_k = min(n_components, len(X) // 25 + 1)
            for k in range(1, max_k+1):
                model = mixture.GaussianMixture(n_components=k, max_iter=gmm_max_iter, n_init=5, random_state=42)
                model.fit(X)
                models.append(model)
            BICs = [model.bic(X) for model in models]
            return models[np.argmin(BICs)]