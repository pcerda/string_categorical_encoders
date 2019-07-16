import os
import sys
import numpy as np
import warnings

from scipy.special import logsumexp
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, \
    LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF, \
    TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.utils import murmurhash3_32, check_random_state

from fastText import load_model
import category_encoders as cat_enc
from dirty_cat import SimilarityEncoder, TargetEncoder
from dirty_cat.similarity_encoder import get_kmeans_prototypes

import gamma_poisson_factorization

CE_HOME = os.environ.get('CE_HOME')
sys.path.append(os.path.abspath(os.path.join(
    CE_HOME, 'python', 'categorical_encoding')))
from get_data import get_data_path


class OneHotEncoderRemoveOne(OneHotEncoder):
    def __init__(self, n_values=None, categorical_features=None,
                 categories='auto', sparse=True, dtype=np.float64,
                 handle_unknown='error'):
        super().__init__()
        self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.n_values = n_values
        self.categorical_features = categorical_features

    def transform(self, X, y=None):
        Xout = super().transform(X)
        return Xout[:, :-1]


class NgramNaiveFisherKernel(SimilarityEncoder):
    """
    Fisher kernel for a simple n-gram probability distribution

    For the moment, the default implementation uses the most-frequent
    prototypes
    """

    def __init__(self, ngram_range=(2, 4),
                 categories='auto', dtype=np.float64,
                 handle_unknown='ignore', hashing_dim=None, n_prototypes=None,
                 random_state=None, n_jobs=None):
        super().__init__()
        self.categories = categories
        self.ngram_range = ngram_range
        self.dtype = np.float64
        self.handle_unknown = handle_unknown
        self.hashing_dim = hashing_dim
        self.n_prototypes = n_prototypes
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        X = self._check_X(X)
        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if ((self.hashing_dim is not None) and
                (not isinstance(self.hashing_dim, int))):
            raise ValueError("value '%r' was specified for hashing_dim, "
                             "which has invalid type, expected None or "
                             "int." % self.hashing_dim)

        if self.categories not in ['auto', 'most_frequent', 'k-means']:
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")

        n_samples, n_features = X.shape
        self.categories_ = list()
        self.random_state_ = check_random_state(self.random_state)

        for i in range(n_features):
            Xi = X[:, i]
            if self.categories == 'auto':
                self.categories_.append(np.unique(Xi))
            elif self.categories == 'most_frequent':
                self.categories_.append(self.get_most_frequent(Xi))
            elif self.categories == 'k-means':
                uniques, count = np.unique(Xi, return_counts=True)
                self.categories_.append(
                    get_kmeans_prototypes(uniques, self.n_prototypes,
                                          sample_weight=count,
                                          random_state=self.random_state_))
            else:
                if self.handle_unknown == 'error':
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                self.categories_.append(np.array(self.categories[i],
                                                 dtype=object))
        return self

    def transform(self, X):
        """Transform X using specified encoding scheme.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_new : 2-d array, shape [n_samples, n_features_new]
            Transformed input.
        """
        X = self._check_X(X)

        n_samples, n_features = X.shape

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)

        min_n, max_n = self.ngram_range

        total_length = sum(len(x) for x in self.categories_)
        X_out = np.empty((len(X), total_length), dtype=self.dtype)
        current_length = 0
        for j, cats in enumerate(self.categories_):
            encoded_Xj = self._ngram_presence_fisher_kernel(X[:, j], cats)
            X_out[:, current_length: current_length + len(cats)] = encoded_Xj
            current_length += len(cats)
        return X_out

    def _ngram_presence_fisher_kernel(self, strings, cats):
        """ given to arrays of strings, returns the
        encoding matrix of size
        len(strings) x len(cats)
        kernel fisher with p
        where p is the presence vector
        """
        unq_strings = np.unique(strings)
        unq_cats, count_j = np.unique(cats, return_counts=True)
        theta = count_j/sum(count_j)
        theta_sum = theta.sum()
        vectorizer = CountVectorizer(analyzer='char',
                                     ngram_range=self.ngram_range)
        Cj = vectorizer.fit_transform(unq_cats)
        Ci = vectorizer.transform(unq_strings)
        m = Cj.shape[1]
        SE_dict = {}
        for i, c_i in enumerate(Ci):
            gamma = np.ones(m) * theta_sum
            similarity = []
            for j, c_j in enumerate(Cj):
                indicator = (c_j != c_i).astype('float64')
                gamma -= indicator * theta[j]
                similarity.append(indicator)
            gamma_inv = 1 / gamma
            del gamma
            similarity = (gamma_inv.sum() -
                          sparse.vstack(similarity).multiply(gamma_inv
                                                             ).sum(axis=1))
            SE_dict[unq_strings[i]] = similarity.reshape(1, -1)
        SE = np.empty((len(strings), len(cats)))
        for i, s in enumerate(strings):
            SE[i, :] = SE_dict[s]
        return np.nan_to_num(SE)

    def _ngram_presence_fisher_kernel2(self, strings, cats):
        """ given to arrays of strings, returns the
        encoding matrix of size
        len(strings) x len(cats)
        kernel fisher with p
        where p is the presence vector
        """
        unq_strings = np.unique(strings)
        unq_cats, count_j = np.unique(cats, return_counts=True)
        theta = count_j/sum(count_j)
        vectorizer = CountVectorizer(analyzer='char',
                                     ngram_range=self.ngram_range,
                                     binary=True)
        Cj = vectorizer.fit_transform(unq_cats)
        Ci = vectorizer.transform(unq_strings)
        m = Cj.shape[1]
        SE_dict = {}
        for i, p_i in enumerate(Ci):
            gamma = np.zeros(m)
            for j, p_j in enumerate(Cj):
                gamma += (p_j == p_i).astype('float64') * theta[j]
            similarity = []
            for j, p_j in enumerate(Cj):
                sim_j = (p_j == p_i).astype('float64') / gamma
                similarity.append(sim_j.sum())
            SE_dict[unq_strings[i]] = np.array(similarity)
        SE = np.empty((len(strings), len(cats)))
        for i, s in enumerate(strings):
            SE[i, :] = SE_dict[s]
        return np.nan_to_num(SE)


class PretrainedFastText(BaseEstimator, TransformerMixin):
    """
    Category embedding using a fastText pretrained model.
    """

    def __init__(self, n_components, language='english'):
        self.n_components = n_components
        self.language = language

    def fit(self, X, y=None):

        path_dict = dict(
            english='crawl-300d-2M-subword.bin',
            french='cc.fr.300.bin',
            hungarian='cc.hu.300.bin')

        if self.language not in path_dict.keys():
            raise AttributeError(
                'language %s has not been downloaded yet' % self.language)

        self.ft_model = load_model(os.path.join(get_data_path(), 'fastText',
                                                path_dict[self.language]))
        return self

    def transform(self, X):
        X = X.ravel()
        unq_X, lookup = np.unique(X, return_inverse=True)
        X_dict = dict()
        for i, x in enumerate(unq_X):
            if x.find('\n') != -1:
                unq_X[i] = ' '.join(x.split('\n'))

        for x in unq_X:
            X_dict[x] = self.ft_model.get_sentence_vector(x)

        X_out = np.empty((len(lookup), 300))
        for x, x_out in zip(unq_X[lookup], X_out):
            x_out[:] = X_dict[x]
        return X_out


class MinHashEncoder(BaseEstimator, TransformerMixin):
    """
    minhash method applied to ngram decomposition of strings
    """

    def __init__(self, n_components, ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.n_components = n_components

    def get_unique_ngrams(self, string, ngram_range):
        """
        Return a list of different n-grams in a string
        """
        spaces = ' '  # * (n // 2 + n % 2)
        string = spaces + " ".join(string.lower().split()) + spaces
        ngram_list = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            string_list = [string[i:] for i in range(n)]
            ngram_list += list(set(zip(*string_list)))
        return ngram_list

    def minhash(self, string, n_components, ngram_range):
        min_hashes = np.ones(n_components) * np.infty
        grams = self.get_unique_ngrams(string, self.ngram_range)
        if len(grams) == 0:
            grams = self.get_unique_ngrams(' Na ', self.ngram_range)
        for gram in grams:
            hash_array = np.array([
                murmurhash3_32(''.join(gram), seed=d, positive=True)
                for d in range(n_components)])
            min_hashes = np.minimum(min_hashes, hash_array)
        return min_hashes/(2**32-1)

    def fit(self, X, y=None):

        self.hash_dict = {}
        for i, x in enumerate(X):
            if x not in self.hash_dict:
                self.hash_dict[x] = self.minhash(
                    x, n_components=self.n_components,
                    ngram_range=self.ngram_range)
        return self

    def transform(self, X):

        X_out = np.zeros((len(X), self.n_components))

        for i, x in enumerate(X):
            if x not in self.hash_dict:
                self.hash_dict[x] = self.minhash(
                    x, n_components=self.n_components,
                    ngram_range=self.ngram_range)

        for i, x in enumerate(X):
            X_out[i, :] = self.hash_dict[x]

        return X_out


class AdHocIndependentPDF(BaseEstimator, TransformerMixin):
    def __init__(self, fisher_kernel=True, dtype=np.float64,
                 ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.count_vectorizer = CountVectorizer(
             analyzer='char', ngram_range=self.ngram_range)
        self.fisher_kernel = fisher_kernel
        self.dtype = dtype

    def fit(self, X, y=None):
        self.cats, self.count = np.unique(X, return_counts=True)
        self.pD = (self.count_vectorizer.fit_transform(self.cats) > 0)
        self.theta = self.count / sum(self.count)
        self.n_features, self.n_vocab = self.pD.shape
        return self

    def transform(self, X):
        unqX = np.unique(X)
        pX = (self.count_vectorizer.transform(unqX) > 0)
        d = len(self.cats)
        encoder_dict = {}
        for i, px in enumerate(pX):
            beta = np.ones((1, self.n_vocab))
            for j, pd in enumerate(self.pD):
                beta -= (px != pd) * self.theta[j]
            inv_beta = 1 / beta
            inv_beta_trans = inv_beta.transpose()
            sum_inv_beta = inv_beta.sum()
            fisher_vector = np.ones((1, d)) * sum_inv_beta
            for j, pd in enumerate(self.pD):
                fisher_vector[0, j] -= (px != pd).dot(inv_beta_trans)
            encoder_dict[unqX[i]] = fisher_vector
        Xout = np.zeros((X.shape[0], d))
        for i, x in enumerate(X):
            Xout[i, :] = encoder_dict[x]
        return np.nan_to_num(Xout).astype(self.dtype)


class NgramsMultinomialMixture(BaseEstimator, TransformerMixin):
    """
    Fisher kernel w/r to the mixture of unigrams model (Nigam, 2000).
    """
    # TODO: add stop_criterion; implement k-means for count-vector;
    # implement version with poisson distribution; add online_method

    def __init__(self, n_topics=10, max_iters=100, fisher_kernel=True,
                 beta_init_type=None, max_mean_change_tol=1e-5,
                 ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.ngrams_count = CountVectorizer(
             analyzer='char', ngram_range=self.ngram_range)
        self.n_topics = n_topics  # parameter k
        self.max_iters = max_iters
        self.fisher_kernel = fisher_kernel
        self.beta_init_type = beta_init_type
        self.max_mean_change_tol = max_mean_change_tol

    def _get_most_frequent(self, X):
        unqX, count = np.unique(X, return_counts=True)
        # assert self.n_topics <= len(unqX)
        count_sort_ind = np.argsort(-count)
        most_frequent_cats = unqX[count_sort_ind][:self.n_topics]
        count_most_frequent = count[count_sort_ind][:self.n_topics]
        return most_frequent_cats, count_most_frequent

    def _max_mean_change(self, last_beta, beta):
        max_mean_change = max(abs((last_beta - beta)).sum(axis=1))
        return max_mean_change

    def _e_step(self, D, unqD, X, unqX, theta, beta):
        log_doc_topic_posterior_dict = {}
        log_fisher_kernel_dict = {}
        for m, d in enumerate(unqD):
            log_P_z_theta = np.log(theta)
            log_beta = np.log(beta)
            log_P_d_zbeta = np.array(
                [d.dot(log_beta[i, :])[0] - 1 for i in range(self.n_topics)])
            log_P_dz_thetabeta = log_P_d_zbeta + log_P_z_theta
            log_doc_topic_posterior_dict[unqX[m]] = (
                log_P_dz_thetabeta - logsumexp(log_P_dz_thetabeta))
            log_fisher_kernel_dict[unqX[m]] = (
                log_P_d_zbeta - logsumexp(log_P_dz_thetabeta))

        log_doc_topic_posterior = np.zeros((D.shape[0], self.n_topics))
        log_fisher_kernel = np.zeros((D.shape[0], self.n_topics))
        for m, x in enumerate(X):
            log_doc_topic_posterior[m, :] = log_doc_topic_posterior_dict[x]
            log_fisher_kernel[m, :] = log_fisher_kernel_dict[x]
        return np.exp(log_doc_topic_posterior), np.exp(log_fisher_kernel)

    def _m_step(self, D, _doc_topic_posterior):
        aux = np.dot(_doc_topic_posterior.transpose(), D.toarray())
        beta = np.divide(1 + aux,
                         np.sum(aux, axis=1).reshape(-1, 1) + self.n_vocab)
        theta = ((1 + np.sum(_doc_topic_posterior, axis=0).reshape(-1)) /
                 (self.n_topics + self.n_samples))
        return theta, beta

    def fit(self, X, y=None):
        unqX = np.unique(X)
        unqD = self.ngrams_count.fit_transform(unqX)
        D = self.ngrams_count.transform(X)
        self.vocabulary = self.ngrams_count.get_feature_names()
        self.n_samples, self.n_vocab = D.shape
        prototype_cats, protoype_counts = self._get_most_frequent(X)
        self.theta_prior = protoype_counts / self.n_topics
        protoD = self.ngrams_count.transform(prototype_cats).toarray() + 1e-5
        if self.beta_init_type == 'most-frequent-categories':
            self.beta_prior = protoD / protoD.sum(axis=1).reshape(-1, 1)
        if self.beta_init_type == 'constant':
            self.beta_prior = (np.ones(protoD.shape) /
                               protoD.sum(axis=1).reshape(-1, 1))
        if self.beta_init_type == 'random':
            np.random.seed(seed=42)
            aux = np.random.uniform(0, 1, protoD.shape) + 1e-5
            self.beta_prior = aux / protoD.sum(axis=1).reshape(-1, 1)

        theta, beta = self.theta_prior, self.beta_prior
        _last_beta = np.zeros((self.n_topics, self.n_vocab))
        for i in range(self.max_iters):
            for i in range(0):
                print(i)
            _doc_topic_posterior, _ = self._e_step(D, unqD, X, unqX,
                                                   theta, beta)
            theta, beta = self._m_step(D, _doc_topic_posterior)
            max_mean_change = self._max_mean_change(_last_beta, beta)
            if max_mean_change < self.max_mean_change_tol:
                print('final n_iters: %d' % i)
                print(max_mean_change)
                break
            _last_beta = beta
        self.theta, self.beta = theta, beta
        return self

    def transform(self, X):
        unqX = np.unique(X)
        unqD = self.ngrams_count.transform(unqX)
        D = self.ngrams_count.transform(X)
        if type(self.fisher_kernel) is not bool:
            raise TypeError('fisher_kernel parameter must be boolean.')
        if self.fisher_kernel is True:
            _, Xout = self._e_step(D, unqD, X, unqX, self.theta, self.beta)
        if self.fisher_kernel is False:
            Xout, _ = self._e_step(D, unqD, X, unqX, self.theta, self.beta)
        return Xout


class AdHocNgramsMultinomialMixture(BaseEstimator, TransformerMixin):
    """
    Fisher kernel w/r to the mixture of unigrams model (Nigam, 2000).
    The dimensionality of the embedding is set to the number of unique
    categories in the training set and the count vector matrix is give
    as initial gues for the parameter beta.
    """

    def __init__(self, n_iters=10, fisher_kernel=True, ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.ngrams_count = CountVectorizer(
             analyzer='char', ngram_range=self.ngram_range)
        self.n_iters = n_iters
        self.fisher_kernel = fisher_kernel

    def _e_step(self, D, unqD, X, unqX, theta, beta):
        doc_topic_posterior_dict = {}
        fisher_kernel_dict = {}
        for m, d in enumerate(unqD):
            P_z_theta = theta
            beta = beta
            P_d_zbeta = np.array(
                [float(d.dot(beta[i, :].transpose()).toarray()) - 1
                 for i in range(self.n_topics)])
            P_dz_thetabeta = P_d_zbeta * P_z_theta
            doc_topic_posterior_dict[unqX[m]] = (
                P_dz_thetabeta / P_dz_thetabeta.sum(axis=0))
            fisher_kernel_dict[unqX[m]] = (
                P_d_zbeta / P_dz_thetabeta.sum(axis=0))

        doc_topic_posterior = np.zeros((D.shape[0], self.n_topics))
        fisher_kernel = np.zeros((D.shape[0], self.n_topics))
        for m, x in enumerate(X):
            doc_topic_posterior[m, :] = doc_topic_posterior_dict[x]
            fisher_kernel[m, :] = fisher_kernel_dict[x]
        return doc_topic_posterior, fisher_kernel

    def _m_step(self, D, _doc_topic_posterior):
        aux = np.dot(_doc_topic_posterior.transpose(), D.toarray())
        beta = np.divide(1 + aux,
                         np.sum(aux, axis=1).reshape(-1, 1) + self.n_vocab)
        theta = ((1 + np.sum(_doc_topic_posterior, axis=0).reshape(-1)) /
                 (self.n_topics + self.n_samples))
        return theta, beta

    def fit(self, X, y=None):
        unqX, self.theta_prior = np.unique(X, return_counts=True)
        self.theta_prior = self.theta_prior/self.theta_prior.sum()
        self.n_topics = len(unqX)
        unqD = self.ngrams_count.fit_transform(unqX)
        D = self.ngrams_count.transform(X)
        self.n_samples, self.n_vocab = D.shape
        self.beta_prior = sparse.csr_matrix(unqD.multiply(1/unqD.sum(axis=1)))
        theta, beta = self.theta_prior, self.beta_prior
        for i in range(self.n_iters):
            _doc_topic_posterior, _ = self._e_step(D, unqD, X, unqX,
                                                   theta, beta)
            theta, beta = self._m_step(D, _doc_topic_posterior)
        self.theta, self.beta = theta, beta
        return self

    def transform(self, X):
        unqX = np.unique(X)
        D = self.ngrams_count.transform(X)
        unqD = self.ngrams_count.transform(unqX)
        if type(self.fisher_kernel) is not bool:
            raise TypeError('fisher_kernel parameter must be boolean.')
        if self.fisher_kernel is True:
            _, Xout = self._e_step(D, unqD, X, unqX, self.theta, self.beta)
        if self.fisher_kernel is False:
            Xout, _ = self._e_step(D, unqD, X, unqX, self.theta, self.beta)
        return Xout


class MDVEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, clf_type):
        self.clf_type = clf_type

    def fit(self, X, y=None):
        if self.clf_type in ['regression']:
            pass
        if self.clf_type in ['binary', 'multiclass']:
            self.classes_ = np.unique(y)
            self.categories_ = np.unique(X)
            self.class_dict = {c: (y == c) for c in self.classes_}
            self.Exy = {x: [] for x in self.categories_}
            X_dict = {x: (X == x) for x in self.categories_}
            for x in self.categories_:
                for j, c in enumerate(self.classes_):
                    aux1 = X_dict[x]
                    aux2 = self.class_dict[c]
                    self.Exy[x].append(np.mean(aux1[aux2]))
        return self

    def transform(self, X):
        if self.clf_type in ['regression']:
            pass
        if self.clf_type in ['binary', 'multiclass']:
            Xout = np.zeros((len(X), len(self.classes_)))
            for i, x in enumerate(X):
                if x in self.Exy:
                    Xout[i, :] = self.Exy[x]
                else:
                    Xout[i, :] = 0
            return Xout


def test_MDVEncoder():
    X_train = np.array(
        ['hola', 'oi', 'bonjour', 'hola', 'oi',
         'hola', 'oi', 'oi', 'hola'])
    y_train = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    X_test = np.array(['hola', 'bonjour', 'hola', 'oi', 'hello'])
    ans = np.array([[2/4, 2/5],
                    [0, 1/5],
                    [2/4, 2/5],
                    [2/4, 2/5],
                    [0, 0]])
    encoder = MDVEncoder(clf_type='binary-clf')
    encoder.fit(X_train, y_train)
    assert np.array_equal(encoder.transform(X_test), ans)


class PasstroughEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, passthrough=True):
        self.passthrough = passthrough

    def fit(self, X, y=None):
        self.encoder = FunctionTransformer(None, validate=True)
        self.encoder.fit(X)
        # self.columns = np.array(X.columns)
        return self

    # def get_feature_names(self):
    #     return self.columns

    def transform(self, X):
        return self.encoder.transform(X)


class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 encoder_name,
                 reduction_method=None,
                 ngram_range=(2, 4),
                 categories='auto',
                 dtype=np.float64,
                 handle_unknown='ignore',
                 clf_type=None,
                 n_components=None):
        self.ngram_range = ngram_range
        self.encoder_name = encoder_name
        self.categories = categories
        self.dtype = dtype
        self.clf_type = clf_type
        self.handle_unknown = handle_unknown
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.encoders_dict = {
            'OneHotEncoder': OneHotEncoder(handle_unknown='ignore'),
            'OneHotEncoder-1': OneHotEncoderRemoveOne(handle_unknown='ignore'),
            'Categorical': None,
            'OneHotEncoderDense': OneHotEncoder(
                handle_unknown='ignore', sparse=False),
            'OneHotEncoderDense-1': OneHotEncoderRemoveOne(
                handle_unknown='ignore', sparse=False),
            'SimilarityEncoder': SimilarityEncoder(
                ngram_range=self.ngram_range, random_state=10),
            'NgramNaiveFisherKernel': NgramNaiveFisherKernel(
                ngram_range=self.ngram_range, random_state=10),
            'ngrams_hot_vectorizer': [],
            'NgramsCountVectorizer': CountVectorizer(
                analyzer='char', ngram_range=self.ngram_range),
            'NgramsTfIdfVectorizer': TfidfVectorizer(
                analyzer='char', ngram_range=self.ngram_range,
                smooth_idf=False),
            'WordNgramsTfIdfVectorizer': TfidfVectorizer(
                analyzer='word', ngram_range=(1, 1),
                smooth_idf=False),
            'TargetEncoder': TargetEncoder(
                clf_type=self.clf_type, handle_unknown='ignore'),
            'MDVEncoder': MDVEncoder(self.clf_type),
            'BackwardDifferenceEncoder': cat_enc.BackwardDifferenceEncoder(),
            'BinaryEncoder': cat_enc.BinaryEncoder(),
            'HashingEncoder': cat_enc.HashingEncoder(),
            'HelmertEncoder': cat_enc.HelmertEncoder(),
            'SumEncoder': cat_enc.SumEncoder(),
            'PolynomialEncoder': cat_enc.PolynomialEncoder(),
            'BaseNEncoder': cat_enc.BaseNEncoder(),
            'LeaveOneOutEncoder': cat_enc.LeaveOneOutEncoder(),
            'NgramsLDA': Pipeline([
                ('ngrams_count',
                 CountVectorizer(
                     analyzer='char', ngram_range=self.ngram_range)),
                ('LDA', LatentDirichletAllocation(
                    n_components=self.n_components, learning_method='batch'),)
                ]),
            'NMF': Pipeline([
                ('ngrams_count',
                 CountVectorizer(
                     analyzer='char', ngram_range=self.ngram_range)),
                ('NMF', NMF(
                    n_components=self.n_components))
                ]),
            'WordNMF': Pipeline([
                ('ngrams_count',
                 CountVectorizer(
                     analyzer='word', ngram_range=(1, 1))),
                ('NMF', NMF(
                    n_components=self.n_components))
                ]),
            'NgramsMultinomialMixture':
                NgramsMultinomialMixture(
                    n_topics=self.n_components, max_iters=10),
            'AdHocNgramsMultinomialMixture':
                AdHocNgramsMultinomialMixture(n_iters=0),
            'AdHocIndependentPDF': AdHocIndependentPDF(),
            'OnlineGammaPoissonFactorization':
                gamma_poisson_factorization.OnlineGammaPoissonFactorization(
                    n_topics=self.n_components, rho=.99, r=None,
                    tol=1e-4, random_state=18, init='k-means++',
                    ngram_range=self.ngram_range,
                    rescale_W=True, max_iter_e_step=10),
            'OnlineGammaPoissonFactorization2':
                gamma_poisson_factorization.OnlineGammaPoissonFactorization(
                    n_topics=self.n_components, r=.3, rho=None,
                    batch_size=256,
                    tol=1e-4, random_state=18, init='k-means++',
                    ngram_range=self.ngram_range,
                    rescale_W=True, max_iter_e_step=20),
            'OnlineGammaPoissonFactorization3':
                gamma_poisson_factorization.OnlineGammaPoissonFactorization(
                    n_topics=self.n_components, r=.3, rho=None,
                    batch_size=256,
                    tol=1e-4, random_state=18, init='k-means',
                    ngram_range=self.ngram_range,
                    rescale_W=True, max_iter_e_step=20),
            'OnlineGammaPoissonFactorization4':
                gamma_poisson_factorization.OnlineGammaPoissonFactorization(
                    n_topics=self.n_components, r=None, rho=.95,
                    batch_size=256,
                    tol=1e-4, random_state=18, init='k-means',
                    ngram_range=self.ngram_range,
                    rescale_W=True, max_iter_e_step=20),
            'WordOnlineGammaPoissonFactorization':
                gamma_poisson_factorization.OnlineGammaPoissonFactorization(
                    n_topics=self.n_components, r=.3,
                    tol=1e-4, random_state=18, init='k-means++',
                    ngram_range=(1, 1), analizer='word',
                    rescale_W=True, max_iter_e_step=10),
            'OnlineGammaPoissonFactorization_fast':
                gamma_poisson_factorization.OnlineGammaPoissonFactorization(
                    n_topics=self.n_components, r=.3, ngram_range=(3, 3),
                    max_iter=1, min_iter=1,
                    tol=1e-4, random_state=18, init='k-means++',
                    rescale_W=False),
            'MinHashEncoder': MinHashEncoder(
                n_components=self.n_components),
            'PretrainedFastText':
                PretrainedFastText(n_components=self.n_components),
            'PretrainedFastText_fr':
                PretrainedFastText(n_components=self.n_components,
                                   language='french'),
            'PretrainedFastText_hu':
                PretrainedFastText(n_components=self.n_components,
                                   language='hungarian'),
            None: FunctionTransformer(None, validate=True),
            'Passthrough': PasstroughEncoder(),
            }
        self.list_1D_array_methods = [
            'NgramsCountVectorizer',
            'NgramsTfIdfVectorizer',
            'WordNgramsTfIdfVectorizer',
            'ngrams_hot_vectorizer',
            'NgramsLDA',
            'NMF',
            'WordNMF',
            'NgramsMultinomialMixture',
            'NgramsMultinomialMixtureKMeans2',
            'AdHocNgramsMultinomialMixture',
            'AdHocIndependentPDF',
            'GammaPoissonFactorization',
            'OnlineGammaPoissonFactorization',
            'WordOnlineGammaPoissonFactorization',
            'OnlineGammaPoissonFactorization2',
            'OnlineGammaPoissonFactorization3',
            'OnlineGammaPoissonFactorization4',
            'OnlineGammaPoissonFactorization_fast',
            'MinHashEncoder',
            'MinMeanMinHashEncoder',
            ]

    def _get_most_frequent(self, X):
        unqX, count = np.unique(X, return_counts=True)
        if self.n_components <= len(unqX):
            warnings.warn(
                'Dimensionality reduction will not be applied because' +
                'the encoding dimension is smaller than the required' +
                'dimensionality: %d instead of %d' %
                (X.shape[1], self.n_components))
            return unqX.ravel()
        else:
            count_sort_ind = np.argsort(-count)
            most_frequent_cats = unqX[count_sort_ind][:self.n_components]
            return np.sort(most_frequent_cats)

    def get_feature_names(self):
        try:
            feature_names = self.encoder.get_feature_names()
        except AttributeError:
            feature_names = self.columns
        return feature_names

    def fit(self, X, y=None):
        assert X.values.ndim == 1
        self.columns = X.name
        X = X.values

        if self.encoder_name not in self.encoders_dict:
            template = ("Encoder %s has not been implemented yet")
            raise ValueError(template % self.encoder_name)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoder_name == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        # if self.reduction_method == 'MostFrequentCategories':
            # unq_cats = self._get_most_frequent(X)
            # _X = []
            # for x in X:
            #     if x in unq_cats:
            #         _X.append(x)
            # X = np.array(_X)
            # del _X

        if self.categories != 'auto':
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")
        self.le = LabelEncoder()

        if self.categories == 'auto':
            self.le.fit(X,)
        else:
            if self.handle_unknown == 'error':
                valid_mask = np.in1d(X, self.categories)
                if not np.all(valid_mask):
                    msg = ("Found unknown categories during fit")
                    raise ValueError(msg)
            self.le.classes_ = np.array(self.categories)

        self.categories_ = self.le.classes_

        n_samples = X.shape[0]
        try:
            self.n_features = X.shape[1]
        except IndexError:
            self.n_features = 1

        if self.encoder_name in self.list_1D_array_methods:
            assert self.n_features == 1
            X = X.reshape(-1)
        else:
            X = X.reshape(n_samples, self.n_features)

        if self.n_features > 1:
            raise ValueError("Encoder does not support more than one feature.")

        self.encoder = self.encoders_dict[self.encoder_name]

        if self.reduction_method == 'most_frequent':
            assert self.n_features == 1
            if len(np.unique(X)) <= self.n_components:
                warnings.warn(
                    'Dimensionality reduction will not be applied because ' +
                    'the encoding dimension is smaller than the required ' +
                    'dimensionality: %d instead of %d' %
                    (len(np.unique(X)), self.n_components))
                self.pipeline = Pipeline([
                    ('encoder', self.encoder)
                    ])
            else:
                self.encoder.categories = 'most_frequent'
                self.encoder.n_prototypes = self.n_components
                self.pipeline = Pipeline([
                    ('encoder', self.encoder)
                    ])
        elif self.reduction_method == 'k-means':
            assert 'SimilarityEncoder' in self.encoder_name
            assert self.n_features == 1
            if len(np.unique(X)) <= self.n_components:
                warnings.warn(
                    'Dimensionality reduction will not be applied because ' +
                    'the encoding dimension is smaller than the required ' +
                    'dimensionality: %d instead of %d' %
                    (len(np.unique(X)), self.n_components))
                self.pipeline = Pipeline([
                    ('encoder', self.encoder)
                    ])
            else:
                self.encoder.categories = 'k-means'
                self.encoder.n_prototypes = self.n_components
                self.pipeline = Pipeline([
                    ('encoder', self.encoder)
                    ])
        elif self.reduction_method is None:
            self.pipeline = Pipeline([
                ('encoder', self.encoder)
                ])
        else:
            self.pipeline = Pipeline([
                ('encoder', self.encoder),
                ('dimension_reduction',
                 DimensionalityReduction(method_name=self.reduction_method,
                                         n_components=self.n_components))
                ])
        # for MostFrequentCategories, change the fit method to consider only
        # the selected categories
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        n_samples = X.shape[0]
        if self.encoder_name in self.list_1D_array_methods:
            pass
        else:
            X = X.values.reshape(n_samples, self.n_features)
        Xout = self.pipeline.transform(X)
        # if Xout.ndim == 1:
        #     Xout.reshape(-1, 1)
        return Xout


class DimensionalityReduction(BaseEstimator, TransformerMixin):
    def __init__(self, method_name=None, n_components=None,
                 column_names=None):
        self.method_name = method_name
        self.n_components = n_components
        self.methods_dict = {
            None: FunctionTransformer(None, accept_sparse=True, validate=True),
            'GaussianRandomProjection': GaussianRandomProjection(
                n_components=self.n_components, random_state=35),
            'TruncatedSVD': TruncatedSVD(
                n_components=self.n_components, random_state=35),
            'most_frequent': 0,
            'k-means': 0,
            'PCA': PCA(n_components=self.n_components, random_state=87)
            }

    def fit(self, X, y=None):
        if self.method_name not in self.methods_dict:
            template = ("Dimensionality reduction method '%s' has not been "
                        "implemented yet")
            raise ValueError(template % self.method_name)

        self.method = self.methods_dict[self.method_name]
        if self.n_components is not None:
            if self.method_name is not None:
                if X.shape[1] <= self.n_components:
                    self.method = self.methods_dict[None]
                    warnings.warn(
                        'Dimensionality reduction will not be applied ' +
                        'because the encoding dimension is smaller than ' +
                        'the required dimensionality: %d instead of %d' %
                        (X.shape[1], self.n_components))

        self.method.fit(X)
        self.n_features = 1
        return self

    def transform(self, X):
        Xout = self.method.transform(X)
        if Xout.ndim == 1:
            return Xout.reshape(-1, 1)
        else:
            return Xout
