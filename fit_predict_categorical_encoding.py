import os
import socket
import time
import datetime
import glob
import json
import warnings

import numpy as np

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_approximation import Nystroem
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn_extra.fast_kernel import FKR_EigenPro, FKC_EigenPro

from joblib import Parallel, delayed

from get_data import Data, get_data_path
from constants import sample_seed, shuffle_seed, clf_seed

from column_encoder import ColumnEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4)


def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def array2list(d):
    """
    For a dictionary d, it transforms tuple/array elements to list elements
    """
    if type(d) is dict:
        for k in d:
            if type(d[k]) is dict:
                d[k] = array2list(d[k])
            elif type(d[k]) in [tuple, np.ndarray]:
                d[k] = list(d[k])
            elif type(d[k]) is list:
                for i, j in enumerate(d[k]):
                    d[k][i] = array2list(d[k][i])
    return d


def method2str(iterable):
    """
    Like array2list but, if there is a method, it transforms it into str.
    """
    d = iterable
    if type(d) is dict:
        for k in d:
            d[k] = method2str(d[k])
    elif type(d) in [tuple, np.ndarray]:
        d = list(d)
        d = method2str(d)
    elif type(d) is list:
        for i, _ in enumerate(d):
            d[i] = method2str(d[i])
    else:
        d = str(d)
    return d


def verify_if_exists(results_path, results_dict):
    results_dict = array2list(results_dict)
    files = glob.glob(os.path.join(results_path, '*.json'))
    # files = [os.path.join(results_path, 'drago2_20170925151218997989.json')]
    for file_ in files:
        data = read_json(file_)
        params_dict = {k: data[k] for k in data
                       if k not in ['results']}
        if params_dict == results_dict:
            return True
    return False


def get_score_metric(clf_type):
    if clf_type == 'regression':
        score_metric = metrics.r2_score
        score_name = 'r2'
    if clf_type == 'binary':
        score_metric = metrics.average_precision_score
        score_name = 'av-prec'
    if clf_type == 'multiclass':
        score_metric = metrics.accuracy_score
        score_name = 'accuracy'
    return score_metric, score_name


def instanciate_estimators(clf_type, classifiers, clf_seed,
                           y=None, **kw):

    score_metric, _ = get_score_metric(clf_type)
    param_grid_LGBM = {
        'learning_rate': [0.1, .05, .5], 'num_leaves': [7, 15, 31]}
    param_grid_XGB = {
        'learning_rate': [0.1, .05, .3], 'max_depth': [3, 6, 9]}
    param_grid_MLP = {
        'learning_rate_init': [.001, .0005, .005],
        'hidden_layer_sizes':
            [(30,), (50,), (100,), (30, 30), (50, 50), (100, 100)]}
    param_grid_EigenProGaussian = {'bandwidth': [1, 5, 25]}
    n_components_eigenpro = 160
    param_grid_nystroem_ridgecv = {
        'kernel_approx__n_components': [1000, 3000],
        'kernel_approx__degree': [2, 3],
    }
    if clf_type == 'binary':
        print(('Fraction by class: True: %0.2f; False: %0.2f'
               % (list(y).count(True) / len(y),
                  list(y).count(False) / len(y))))
        cw = 'balanced'
        clfs = {
            'L2RegularizedLinearModel':
                linear_model.LogisticRegressionCV(
                    class_weight=cw, max_iter=100, solver='sag',
                    penalty='l2', n_jobs=1, cv=3, multi_class='multinomial'),
            'GradientBoosting':
                ensemble.GradientBoostingClassifier(n_estimators=100),
            'LGBM':
                GridSearchCV(
                    estimator=LGBMClassifier(
                        n_estimators=100, n_jobs=1, is_unbalance=True),
                    param_grid=param_grid_LGBM, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'XGB':
                GridSearchCV(
                    estimator=XGBClassifier(
                        n_estimators=100, n_jobs=1),
                    param_grid=param_grid_XGB, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'MLP':
                MLPClassifier(
                    hidden_layer_sizes=(30, 30), activation='relu',
                    solver='adam', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=0.001,
                    power_t=0.5, max_iter=200, shuffle=True,
                    random_state=None, tol=0.0001, verbose=False,
                    warm_start=False, momentum=0.9, nesterovs_momentum=True,
                    early_stopping=False, validation_fraction=0.1,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                    n_iter_no_change=10),
            'MLPGridSearchCV':
                GridSearchCV(
                    estimator=MLPClassifier(
                        hidden_layer_sizes=(30, 30), activation='relu',
                        solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='adaptive', learning_rate_init=0.001,
                        power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False,
                        warm_start=False, momentum=0.9,
                        nesterovs_momentum=True,
                        early_stopping=False, validation_fraction=0.1,
                        beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                        n_iter_no_change=10),
                    param_grid=param_grid_MLP, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'EigenProPolynomial':
                FKC_EigenPro(
                    batch_size="auto", n_epoch=10,
                    n_components=n_components_eigenpro,
                    subsample_size="auto", kernel="polynomial",
                    bandwidth=5, gamma=None, degree=2, coef0=1,
                    kernel_params=None, random_state=None),
            'EigenProGaussian160':
                GridSearchCV(
                    estimator=FKC_EigenPro(
                        batch_size="auto", n_epoch=10,
                        n_components=n_components_eigenpro,
                        subsample_size="auto", kernel="gaussian",
                        gamma=None, degree=2, coef0=1,
                        kernel_params=None, random_state=None),
                    param_grid=param_grid_EigenProGaussian, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'EigenProGaussian1000':
                GridSearchCV(
                    estimator=FKC_EigenPro(
                        batch_size="auto", n_epoch=10,
                        n_components=1000,
                        subsample_size="auto", kernel="gaussian",
                        gamma=None, degree=2, coef0=1,
                        kernel_params=None, random_state=None),
                    param_grid=param_grid_EigenProGaussian, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'NystroemRidgeCV':
                GridSearchCV(
                    estimator=Pipeline([
                        ('kernel_approx',
                         Nystroem(kernel="polynomial",
                                  n_components=None,
                                  random_state=clf_seed, degree=2)),
                        ('classifier',
                         linear_model.LogisticRegressionCV(
                             class_weight=cw, max_iter=100, solver='sag',
                             penalty='l2', n_jobs=1, cv=3,
                             multi_class='multinomial'))]),
                    param_grid=param_grid_nystroem_ridgecv, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            }

    elif clf_type == 'multiclass':
        print('fraction of the most frequent class:',
              max([list(y).count(x) for x in set(list(y))]) / len(list(y)))
        clfs = {
            'L2RegularizedLinearModel':
                linear_model.LogisticRegressionCV(
                    penalty='l2', n_jobs=1, cv=3, multi_class='multinomial',
                    solver='sag', max_iter=100),
            'GradientBoosting':
                ensemble.GradientBoostingClassifier(n_estimators=100),
            'LGBM':
                GridSearchCV(
                    estimator=LGBMClassifier(
                        n_estimators=100, n_jobs=1),
                    param_grid=param_grid_LGBM, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'XGB':
                GridSearchCV(
                    estimator=XGBClassifier(
                        n_estimators=100, n_jobs=1, objective='multi:softmax',
                        num_class=len(np.unique(y))),
                    param_grid=param_grid_XGB, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'MLP':
                MLPClassifier(
                    hidden_layer_sizes=(30, 30), activation='relu',
                    solver='adam', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=0.001,
                    power_t=0.5, max_iter=200, shuffle=True,
                    random_state=None, tol=0.0001, verbose=False,
                    warm_start=False, momentum=0.9, nesterovs_momentum=True,
                    early_stopping=False, validation_fraction=0.1,
                    beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                    n_iter_no_change=10),
            'MLPGridSearchCV':
                GridSearchCV(
                    estimator=MLPClassifier(
                        hidden_layer_sizes=(30, 30), activation='relu',
                        solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='adaptive', learning_rate_init=0.001,
                        power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None, tol=0.0001, verbose=False,
                        warm_start=False, momentum=0.9,
                        nesterovs_momentum=True,
                        early_stopping=False, validation_fraction=0.1,
                        beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                        n_iter_no_change=10),
                    param_grid=param_grid_MLP, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'EigenProPolynomial':
                FKC_EigenPro(
                    batch_size="auto", n_epoch=10,
                    n_components=n_components_eigenpro,
                    subsample_size="auto", kernel="polynomial",
                    gamma=None, degree=2, coef0=1,
                    kernel_params=None, random_state=None),
            'EigenProGaussian160':
                GridSearchCV(
                    estimator=FKC_EigenPro(
                        batch_size="auto", n_epoch=10,
                        n_components=n_components_eigenpro,
                        subsample_size="auto", kernel="gaussian",
                        gamma=None, degree=2, coef0=1,
                        kernel_params=None, random_state=None),
                    param_grid=param_grid_EigenProGaussian, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'EigenProGaussian1000':
                GridSearchCV(
                    estimator=FKC_EigenPro(
                        batch_size="auto", n_epoch=10,
                        n_components=1000,
                        subsample_size="auto", kernel="gaussian",
                        gamma=None, degree=2, coef0=1,
                        kernel_params=None, random_state=None),
                    param_grid=param_grid_EigenProGaussian, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'NystroemRidgeCV':
                GridSearchCV(
                    estimator=Pipeline([
                        ('kernel_approx',
                         Nystroem(kernel="polynomial",
                                  n_components=None,
                                  random_state=clf_seed, degree=2)),
                        ('classifier',
                         linear_model.LogisticRegressionCV(
                             penalty='l2', n_jobs=1, cv=3,
                             multi_class='multinomial',
                             solver='sag', max_iter=100))]),
                    param_grid=param_grid_nystroem_ridgecv, cv=3,
                    scoring=metrics.make_scorer(score_metric)),

            }
    elif clf_type == 'regression':
        clfs = {
            'L2RegularizedLinearModel':
                linear_model.RidgeCV(cv=3),
            'GradientBoosting':
                ensemble.GradientBoostingRegressor(n_estimators=100),
            'LGBM':
                GridSearchCV(
                    estimator=LGBMRegressor(
                        n_estimators=100, n_jobs=1),
                    param_grid=param_grid_LGBM, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'XGB':
                GridSearchCV(
                    estimator=XGBRegressor(
                        n_estimators=100, n_jobs=1),
                    param_grid=param_grid_XGB, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'MLP':
                MLPRegressor(
                    hidden_layer_sizes=(30, 30), activation='relu',
                    solver='adam', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=0.001,
                    power_t=0.5, max_iter=200, shuffle=True, random_state=None,
                    tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                    nesterovs_momentum=True, early_stopping=False,
                    validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, n_iter_no_change=10),
            'MLPGridSearchCV':
                GridSearchCV(
                    estimator=MLPRegressor(
                        hidden_layer_sizes=(30, 30), activation='relu',
                        solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        power_t=0.5, max_iter=200, shuffle=True,
                        random_state=None,
                        tol=0.0001, verbose=False, warm_start=False,
                        momentum=0.9,
                        nesterovs_momentum=True, early_stopping=False,
                        validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                        epsilon=1e-08, n_iter_no_change=10),
                    param_grid=param_grid_MLP, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'EigenProPolynomial':
                FKR_EigenPro(
                    batch_size="auto", n_epoch=10,
                    n_components=n_components_eigenpro,
                    subsample_size="auto", kernel="polynomial",
                    bandwidth=5, gamma=None, degree=2, coef0=1,
                    kernel_params=None, random_state=None),
            'EigenProGaussian160':
                GridSearchCV(
                    estimator=FKR_EigenPro(
                        batch_size="auto", n_epoch=10,
                        n_components=n_components_eigenpro,
                        subsample_size="auto", kernel="gaussian",
                        gamma=None, degree=2, coef0=1,
                        kernel_params=None, random_state=None),
                    param_grid=param_grid_EigenProGaussian, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'EigenProGaussian1000':
                GridSearchCV(
                    estimator=FKR_EigenPro(
                        batch_size="auto", n_epoch=10,
                        n_components=1000,
                        subsample_size="auto", kernel="gaussian",
                        gamma=None, degree=2, coef0=1,
                        kernel_params=None, random_state=None),
                    param_grid=param_grid_EigenProGaussian, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            'NystroemRidgeCV':
                GridSearchCV(
                    estimator=Pipeline([
                        ('kernel_approx',
                         Nystroem(kernel="polynomial",
                                  n_components=None,
                                  random_state=clf_seed, degree=2)),
                        ('classifier',
                         linear_model.RidgeCV(cv=3))]),
                    param_grid=param_grid_nystroem_ridgecv, cv=3,
                    scoring=metrics.make_scorer(score_metric)),
            }
    else:
        raise ValueError("{} not recognized".format(clf_type))

    clfs = [clfs[clf] for clf in classifiers]
    for clf in clfs:
        try:
            if 'random_state' in clf.estimator.get_params():
                clf.estimator.set_params(random_state=clf_seed)
        except AttributeError:
            if 'random_state' in clf.get_params():
                clf.set_params(random_state=clf_seed)
    return clfs


def select_cross_val(clf_type, n_splits, test_size, custom_cv=None,
                     col_name=None, random_state=shuffle_seed):
    if custom_cv is None:
        if clf_type in ['regression']:
            cv = ShuffleSplit(n_splits=n_splits,
                              test_size=test_size,
                              random_state=random_state)
        if clf_type in ['binary', 'multiclass']:
            cv = StratifiedShuffleSplit(n_splits=n_splits,
                                        test_size=test_size,
                                        random_state=random_state)
        return cv
    else:
        assert custom_cv.__class__.__name__ == 'LeaveRareCatsOut'
        custom_cv.n_splits = n_splits
        custom_cv.test_size = test_size
        custom_cv.random_state = shuffle_seed
        custom_cv.col_name = col_name
        return custom_cv


def select_scaler():
    scaler = preprocessing.StandardScaler(with_mean=False)
    return scaler


def choose_nrows(dataset_name):
    if dataset_name in [
            'open_payments',
            'traffic_violations',
            'federal_election',
            'medical_charge',
            'beer_reviews',
            'road_safety',
            'public_procurement',
            'crime_data',
            'met_objects',
            'drug_directory',
            'consumer_complaints',
            'intrusion_detection',
            'kickstarter_projects',
            'building_permits',
            'wine_reviews',
            'firefighter_interventions',
            ]:
        n_rows = 100000
    else:
        n_rows = -1
    return n_rows


def get_column_action(col_action, xcols, encoder, reduction_method,
                      n_components, clf_type):
    encoder_type = {
        'OneHotEncoderDense':
            ColumnEncoder(encoder_name='OneHotEncoderDense',
                          clf_type=clf_type),
        'OneHotEncoderDense-1':
            ColumnEncoder(encoder_name='OneHotEncoderDense-1',
                          clf_type=clf_type),
        # 'OneHotEncoder-1' has to be implemented
        'TargetEncoder':
            ColumnEncoder(encoder_name='TargetEncoder',
                          clf_type=clf_type, handle_unknown='ignore'),
        'Special':
            ColumnEncoder(encoder_name=encoder,
                          clf_type=clf_type,
                          reduction_method=reduction_method,
                          n_components=n_components),
        # 'Numerical':
        #     ColumnEncoder(encoder_name=None, clf_type=clf_type),
        'Numerical':
            ColumnEncoder(encoder_name='Passthrough', clf_type=clf_type)
        }

    column_action = {col: encoder_type[col_action[col]]
                     for col in xcols}
    # if verbose:
    #     print('Column action')
    #     for col in column_action:
    #         print('\t', col, column_action[col])
    return column_action


def fit_predict_fold(data, scaler, column_action, clf, encoder,
                     reduction_method, n_components,
                     fold, n_splits, train_index, test_index):
    """
    fits and predicts a X with y given multiple parameters.
    """
    start_encoding = time.time()
    y = data.df[data.ycol].values
    data_train = data.df.iloc[train_index, :]
    y_train = y[train_index]

    # Use ColumnTransformer to combine the features
    transformer = ColumnTransformer([(col, column_action[col], col)
                                     for col in data.xcols])

    X_train = transformer.fit_transform(data_train[data.xcols], y_train)
    X_train = scaler.fit_transform(X_train, y_train)
    encoding_time = time.time() - start_encoding

    start_training = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_training

    train_shape = X_train.shape
    del X_train

    data_test = data.df.iloc[test_index, :]
    y_test = y[test_index]
    X_test = transformer.transform(data_test)
    del transformer
    X_test = scaler.transform(X_test)

    score_metric, score_name = get_score_metric(data.clf_type)

    if data.clf_type in ['regression', 'multiclass']:
        y_pred = clf.predict(X_test)
    elif data.clf_type == 'binary':
        try:
            y_pred = clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_pred = clf.decision_function(X_test)

    score = score_metric(y_test, y_pred)

    print('%s (%d/%d), ' % (data.name, fold, n_splits),
          'encoder: %s, ' % encoder,
          'n_samp: %d, ' % train_shape[0],
          'n_feat: %d, ' % train_shape[1],
          '%s: %.4f, ' % (score_name, score),
          'enc-time: %.0f s.' % encoding_time,
          'train-time: %.0f s.' % training_time)
    results = [fold, y_train.shape[0], X_test.shape[1],
               score, encoding_time, training_time]
    return results


def fit_predict_categorical_encoding(datasets, str_preprocess, encoders,
                                     classifiers, reduction_methods,
                                     n_components,
                                     test_size, n_splits, n_jobs, results_path,
                                     model_path=None, custom_cv=None):
    '''
    Learning with dirty categorical variables.
    '''
    path = get_data_path()
    results_path = os.path.join(path, results_path)
    model_path = os.path.join(path, model_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for dataset in datasets:
        n_rows = choose_nrows(dataset_name=dataset)
        for encoder in encoders:
            print('Dataset: %s' % dataset)
            data = Data(dataset).get_df()
            data.preprocess(n_rows=n_rows, str_preprocess=str_preprocess)
            special_col = [col for col in data.col_action
                           if data.col_action[col] == 'Special'][0]
            if type(encoder) is list:
                # special_col = [col for col in data.col_action
                #                if data.col_action[col] == 'Special'][0]
                for i, enc in enumerate(encoder):
                    print(enc)
                    if i == 0:
                        data.col_action[special_col] = 'Special'
                    else:
                        new_col = '%s_%d' % (special_col, i)
                        data.df[new_col] = data.df[special_col].copy()
                        data.col_action[new_col] = enc
                        data.xcols.append(new_col)
            for reduction_method in reduction_methods:
                print('Data shape: %d, %d' % data.df.shape)
                cv = select_cross_val(data.clf_type, n_splits, test_size,
                                      custom_cv=custom_cv,
                                      col_name=special_col)
                scaler = select_scaler()

                # Define classifiers
                clfs = instanciate_estimators(
                    data.clf_type, classifiers, clf_seed,
                    y=data.df.loc[:, data.ycol].values,
                    model_path=model_path)

                for i, clf in enumerate(clfs):
                    print(
                        '{}: {} \n{}: {} \n{}: {} \n{}: {} \n{}: {},{}'.format(
                            'Prediction column', data.ycol,
                            'Task type', str(data.clf_type),
                            'Classifier', clf,
                            'Encoder', encoder,
                            'Dimension reduction', reduction_method,
                            n_components))

                    try:
                        clf_name = clf.estimator.__class__.__name__
                        results_dict = {
                            'dataset': data.name,
                            'n_splits': n_splits,
                            'test_size': test_size,
                            'n_rows': n_rows,
                            'encoder': encoder,
                            'str_preprocess': str_preprocess,
                            'clf': [classifiers[i], clf_name,
                                    clf.estimator.get_params()],
                            'ShuffleSplit': [cv.__class__.__name__],
                            'scaler': [scaler.__class__.__name__,
                                       scaler.get_params()],
                            'sample_seed': sample_seed,
                            'shuffleseed': shuffle_seed,
                            'col_action': data.col_action,
                            'clf_type': data.clf_type,
                            'dimension_reduction': [reduction_method,
                                                    n_components]
                            }
                    except AttributeError:
                        clf_name = clf.__class__.__name__
                        results_dict = {
                            'dataset': data.name,
                            'n_splits': n_splits,
                            'test_size': test_size,
                            'n_rows': n_rows,
                            'encoder': encoder,
                            'str_preprocess': str_preprocess,
                            'clf': [classifiers[i], clf_name,
                                    clf.get_params()],
                            'ShuffleSplit': [cv.__class__.__name__],
                            'scaler': [scaler.__class__.__name__,
                                       scaler.get_params()],
                            'sample_seed': sample_seed,
                            'shuffleseed': shuffle_seed,
                            'col_action': data.col_action,
                            'clf_type': data.clf_type,
                            'dimension_reduction': [reduction_method,
                                                    n_components]
                            }

                    if verify_if_exists(results_path, results_dict):
                        print('Prediction already exists.\n')
                        continue

                    start = time.time()
                    if type(encoder) is str:
                        column_action = get_column_action(
                            data.col_action, data.xcols, encoder,
                            reduction_method, n_components, data.clf_type)
                    if type(encoder) is list:
                        column_action = get_column_action(
                            data.col_action, data.xcols, encoder[0],
                            reduction_method, n_components, data.clf_type)
                    pred = Parallel(n_jobs=n_jobs)(
                        delayed(fit_predict_fold)(
                            data, scaler, column_action, clf, encoder,
                            reduction_method, n_components,
                            fold, cv.n_splits, train_index, test_index)
                        for fold, (train_index, test_index)
                        in enumerate(
                            cv.split(data.df, data.df[data.ycol].values)))
                    pred = np.array(pred)
                    results = {'fold': list(pred[:, 0]),
                               'n_train_samples': list(pred[:, 1]),
                               'n_train_features': list(pred[:, 2]),
                               'score': list(pred[:, 3]),
                               'encoding_time': list(pred[:, 4]),
                               'training_time': list(pred[:, 5])}
                    results_dict['results'] = results

                    # Saving results
                    pc_name = socket.gethostname()
                    now = ''.join([c for c in str(datetime.datetime.now())
                                   if c.isdigit()])
                    filename = ('%s_%s_%s_%s_%s.json' %
                                (pc_name, data.name, classifiers[i],
                                 encoder, now))
                    results_file = os.path.join(results_path, filename)
                    results_dict = array2list(results_dict)

                    # patch for nystrom + ridge
                    if clf.__class__.__name__ == 'GridSearchCV':
                        if clf.estimator.__class__.__name__ == 'Pipeline':
                            results_dict['clf'] = method2str(
                                results_dict['clf'])

                    write_json(results_dict, results_file)
                    print('prediction time: %.1f s.' % (time.time() - start))
                    print('Saving results to: %s\n' % results_file)
