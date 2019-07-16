import os
import sys

CE_HOME = os.environ.get('CE_HOME')
sys.path.append(os.path.abspath(os.path.join(
    CE_HOME, 'python', 'categorical_encoding')))
from fit_predict_categorical_encoding import fit_predict_categorical_encoding
from leave_rare_cats_out import LeaveRareCatsOut
from get_data import get_data_path

'''
Learning with dirty categorical variables.
'''

# Parameters ##################################################################
datasets = [
    'vancouver_employee',
    'journal_influence',
    'colleges',
    'midwest_survey',
    'employee_salaries',
    'medical_charge',
    'kickstarter_projects',
    'crime_data',
    'open_payments',
    'traffic_violations',
    'federal_election',
    'public_procurement',
    'building_permits',
    'road_safety',
    'met_objects',
    'drug_directory',
    'wine_reviews',
    ]
n_jobs = 20
n_splits = 20
test_size = 1./3
str_preprocess = True
n_components = 100
results_path = os.path.join(get_data_path(), 'results', 'jmlr2019_2')
# results_path = os.path.join(get_data_folder(), 'results',
#                             'kdd_2019_only_cats')
classifiers = [
    # 'NystroemRidgeCV',
    # 'L2RegularizedLinearModel',
    # 'EigenProGaussian160',
    # 'EigenProPolynomial',
    # 'XGB',
    # 'LGBM',
    # 'KNN',
    'MLPGridSearchCV',
    ]
###############################################################################

# Probabilistic topic models without dimensionality reduction #################
encoders = [
    ## 'MinHashEncoder',
    ## 'OnlineGammaPoissonFactorization3',
    # 'WordOnlineGammaPoissonFactorization',
    # 'NMF',
    # 'WordNMF',
    # 'TargetEncoder'
    ]
reduction_methods = [None]
fit_predict_categorical_encoding(datasets=datasets,
                                 str_preprocess=str_preprocess,
                                 encoders=encoders, classifiers=classifiers,
                                 reduction_methods=reduction_methods,
                                 n_components=n_components,
                                 test_size=test_size, n_splits=n_splits,
                                 n_jobs=n_jobs, results_path=results_path,
                                 model_path='')

# Encoders with TruncatedSVD ##################################################
encoders = [
    'PretrainedFastText',
    # 'PretrainedFastText_hu',
    # 'PretrainedFastText_fr',
    'OneHotEncoder',
    'NgramsCountVectorizer',
    'NgramsTfIdfVectorizer',
    # 'WordNgramsTfIdfVectorizer',
    ]
reduction_methods = ['TruncatedSVD']
fit_predict_categorical_encoding(datasets=datasets,
                                 str_preprocess=str_preprocess,
                                 encoders=encoders, classifiers=classifiers,
                                 reduction_methods=reduction_methods,
                                 n_components=n_components,
                                 test_size=test_size, n_splits=n_splits,
                                 n_jobs=n_jobs, results_path=results_path,
                                 model_path='')

# Encoders with most frequent protoypes #######################################
encoders = [
    # 'SimilarityEncoder',
    # 'NgramNaiveFisherKernel',
    ]
reduction_methods = ['most_frequent']
# '-', 'GaussianRandomProjection', 'most_frequent', 'k-means', 'TruncatedSVD'
fit_predict_categorical_encoding(datasets=datasets,
                                 str_preprocess=str_preprocess,
                                 encoders=encoders, classifiers=classifiers,
                                 reduction_methods=reduction_methods,
                                 n_components=n_components,
                                 test_size=test_size, n_splits=n_splits,
                                 n_jobs=n_jobs, results_path=results_path,
                                 model_path='')

# Encoders with k-means protoypes #############################################
encoders = [
    ## 'SimilarityEncoder',
    ]
reduction_methods = ['k-means']
fit_predict_categorical_encoding(datasets=datasets,
                                 str_preprocess=str_preprocess,
                                 encoders=encoders, classifiers=classifiers,
                                 reduction_methods=reduction_methods,
                                 n_components=n_components,
                                 test_size=test_size, n_splits=n_splits,
                                 n_jobs=n_jobs, results_path=results_path,
                                 model_path='')

# Encoders with GaussianRandomProjection ######################################
encoders = [
    # 'SimilarityEncoder',
    ## 'PretrainedFastText',
    # 'PretrainedFastText_hu',
    # 'PretrainedFastText_fr',
    ## 'OneHotEncoder',
    ## 'NgramsCountVectorizer',
    ## 'NgramsTfIdfVectorizer',
    ]
reduction_methods = ['GaussianRandomProjection']
fit_predict_categorical_encoding(datasets=datasets,
                                 str_preprocess=str_preprocess,
                                 encoders=encoders, classifiers=classifiers,
                                 reduction_methods=reduction_methods,
                                 n_components=n_components,
                                 test_size=test_size, n_splits=n_splits,
                                 n_jobs=n_jobs, results_path=results_path,
                                 model_path='')
