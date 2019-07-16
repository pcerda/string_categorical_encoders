import os
import pandas as pd
import numpy as np
import glob
import json
import socket
import warnings

from sklearn.preprocessing import LabelEncoder

from constants import sample_seed
import datasets.src

CE_HOME = os.environ.get('CE_HOME')
dataset_config_file = os.path.join(CE_HOME, 'python', 'categorical_encoding',
                                   'datasets_config.json')
with open(dataset_config_file, 'r') as f:
    DATASET_CONFIG = json.load(f)


def preprocess_data(df, cols):
    def string_normalize(s):
        res = str(s).lower()
        # res = ' ' + res + ' '
        return res

    for col in cols:
        print('Preprocessing column: %s' % col)
        df[col] = [string_normalize(str(r)) for r in df[col]]
    return df


def get_data_path():
    hostname = socket.gethostname()
    if hostname in ['drago2', 'drago3', 'drago4']:
        path = '/storage/inria/pcerda/data'
    elif hostname in ['paradox', 'paradigm', 'parametric', 'parabolic']:
        path = '/storage/local/pcerda/data'
    elif hostname in ['drago']:
        path = '/storage/workspace/pcerda/data'
    else:
        path = os.path.join(CE_HOME, 'data')
    return path


def create_folder(path, folder):
    if not os.path.exists(os.path.join(path, folder)):
        os.makedirs(os.path.join(path, folder))
    return


def print_unique_values(df):
    for col in df.columns:
        print(col, df[col].unique().shape)
        print(df[col].unique())
        print('\n')


def check_nan_percentage(df, col_name):
    threshold = .15
    missing_fraction = df[col_name].isnull().sum()/df.shape[0]
    if missing_fraction > threshold:
        warnings.warn(
            "Fraction of missing values for column '%s' "
            "(%.3f) is higher than %.3f"
            % (col_name, missing_fraction, threshold))


class Data:
    ''' Given the dataset name, return the respective dataframe as other
    relevant information.'''

    def __init__(self, name):
        self.name = name
        self.configs = None
        self.xcols, self.ycol = None, None
        self.col_action = DATASET_CONFIG[self.name]["col_action"]
        self.clf_type = DATASET_CONFIG[self.name]["clf_type"]
        self.dirtiness_type = DATASET_CONFIG[self.name]["dirtiness_type"]
        self.datasets_with_downloader = [
            'employee_salaries',
            'journal_influence',
            'met_objects',
            'colleges',
            'beer_reviews',
            'midwest_survey',
            'medical_charge',
            'traffic_violations',
            'crime_data',
            'adult',
            'public_procurement',
            'intrusion_detection',
            'emobank',
            'text_emotion',
            ]
        self.has_downloader = False
        if self.name in self.datasets_with_downloader:
            self.has_downloader = True

        if name == 'indultos_espana':
            '''Source: '''
            data_path = os.path.join(get_data_path(),
                                     'bigml/Indultos_en_Espana_1996-2013')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Indultos_en_Espana_1996-2013.csv')

        if name == 'open_payments':
            '''Source: '''
            data_path = os.path.join(get_data_path(), 'docs_payments')
            data_file = os.path.join(data_path, 'output', 'DfD.csv')

        if name == 'road_safety':
            '''Source: https://data.gov.uk/dataset/road-accidents-safety-data
            '''
            data_path = os.path.join(get_data_path(), 'road_safety')
            create_folder(data_path, 'output/results')
            data_file = [os.path.join(data_path, 'raw', '2015_Make_Model.csv'),
                         os.path.join(data_path, 'raw', 'Accidents_2015.csv'),
                         os.path.join(data_path, 'raw', 'Casualties_2015.csv'),
                         os.path.join(data_path, 'raw', 'Vehicles_2015.csv')]

        if name == 'consumer_complaints':
            '''Source: https://catalog.data.gov/dataset/
                       consumer-complaint-database
               Documentation: https://cfpb.github.io/api/ccdb//fields.html'''
            data_path = os.path.join(get_data_path(), 'consumer_complaints')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Consumer_Complaints.csv')

        if name == 'product_relevance':
            '''Source: '''
            data_path = os.path.join(get_data_path(), 'product_relevance')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'train.csv')

        if name == 'federal_election':
            '''Source: https://classic.fec.gov/finance/disclosure/
                       ftpdet.shtml#a2011_2012'''
            data_path = os.path.join(get_data_path(), 'federal_election')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'itcont.txt')
            self.data_dict_file = os.path.join(data_path, 'data_dict.csv')

        if name == 'drug_directory':
            '''Source:
            https://www.fda.gov/Drugs/InformationOnDrugs/ucm142438.htm'''
            data_path = os.path.join(get_data_path(), 'drug_directory')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'product.txt')

        if name == 'french_companies':
            '''Source: '''
            data_path = os.path.join(get_data_path(), 'french_companies')
            create_folder(data_path, 'output/results')
            data_file = [os.path.join(data_path, 'raw',
                                      'datasets', 'chiffres-cles-2017.csv'),
                         os.path.join(data_path, 'raw',
                                      'datasets', 'avis_attribution_2017.csv')]

        if name == 'dating_profiles':
            '''Source: https://github.com/rudeboybert/JSE_OkCupid'''
            data_path = os.path.join(get_data_path(), 'dating_profiles')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'profiles.csv')

        if name == 'cacao_flavors':
            '''Source: https://www.kaggle.com/rtatman/chocolate-bar-ratings/'''
            data_path = os.path.join(get_data_path(), 'cacao_flavors')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'flavors_of_cacao.csv')

        if name == 'wine_reviews':
            '''Source: https://www.kaggle.com/zynicide/wine-reviews/home'''
            data_path = os.path.join(get_data_path(), 'wine_reviews')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw', 'winemag-data_first150k.csv')

        if name == 'house_prices':
            '''Source: https://www.kaggle.com/c/
            house-prices-advanced-regression-techniques/data'''
            data_path = os.path.join(get_data_path(), 'house_prices')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'train.csv')

        if name == 'kickstarter_projects':
            '''Source: https://www.kaggle.com/kemical/kickstarter-projects'''
            data_path = os.path.join(get_data_path(), 'kickstarter_projects')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw', 'ks-projects-201612.csv')

        if name == 'building_permits':
            '''Source:
            https://www.kaggle.com/chicago/chicago-building-permits'''
            data_path = os.path.join(get_data_path(), 'building_permits')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'building-permits.csv')

        if name == 'california_housing':
            '''Source:
            https://github.com/ageron/handson-ml/tree/master/datasets/housing
            '''
            data_path = os.path.join(get_data_path(), 'california_housing')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'housing.csv')

        if name == 'house_sales':
            '''Source: https://www.kaggle.com/harlfoxem/housesalesprediction'''
            data_path = os.path.join(get_data_path(), 'house_sales')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'kc_house_data.csv')

        if name == 'vancouver_employee':
            '''Source: https://data.vancouver.ca/datacatalogue/
            employeeRemunerationExpensesOver75k.htm

            Remuneration and Expenses for Employees Earning over $75,000
            '''
            data_path = os.path.join(get_data_path(), 'vancouver_employee')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw',
                '2017StaffRemunerationOver75KWithExpenses.csv')

        if name == 'firefighter_interventions':
            '''Source:
            https://www.data.gouv.fr/fr/datasets/interventions-des-pompiers/
            '''
            data_path = os.path.join(
                get_data_path(), 'firefighter_interventions')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(
                data_path, 'raw', 'interventions-hebdo-2010-2017.csv')

        if name == 'criteo':
            '''Source:
            https://labs.criteo.com/2014/08/criteo-release-public-datasets/
            '''
            data_path = os.path.join(
                get_data_path(), 'criteo')
            data_file = os.path.join(
                data_path, 'raw', 'data.txt')

        # add here the path to a new dataset ##################################
        # if name == 'new_dataset':
        #     '''Source: '''
        #     data_path = os.path.join(get_data_path(), 'new_dataset')
        #     create_folder(data_path, 'output/results')
        #     data_file = os.path.join(data_path, 'raw', 'data_file.csv')
        #######################################################################
        if self.name not in self.datasets_with_downloader:
            self.file = data_file
            self.path = data_path

    def preprocess(self, n_rows=-1, str_preprocess=True,
                   clf_type=None):
        self.col_action = {k: v for k, v in self.col_action.items()
                           if v != 'Delete'}
        self.xcols = [key for key in self.col_action
                      if self.col_action[key] is not 'y']
        self.ycol = [key for key in self.col_action
                     if self.col_action[key] is 'y'][0]
        for col in self.col_action:
            check_nan_percentage(self.df, col)
            if self.col_action[col] in ['OneHotEncoderDense', 'OneHotEncoder',
                                        'Special', 'OneHotEncoderDense-1']:
                self.df[col] = self.df[col].astype(str)
                self.df = self.df.fillna(value={col: 'na'})
        self.df = self.df.dropna(
            axis=0, subset=[c for c in self.xcols if self.col_action[c]
                            is not 'Delete'] + [self.ycol])

        if n_rows == -1:
            self.df = self.df.sample(
                frac=1, random_state=sample_seed).reset_index(drop=True)
        else:
            self.df = self.df.sample(
                n=n_rows, random_state=sample_seed).reset_index(drop=True)
        if str_preprocess:
            self.df = preprocess_data(
                self.df, [key for key in self.col_action
                          if self.col_action[key] == 'Special'])

        # label encoder for the target variable
        if self.clf_type in ['binary', 'multiclass']:
            value_counts = self.df[self.ycol].value_counts()
            le = LabelEncoder()
            le.fit(value_counts.index)
            self.df['target_count'] = self.df[self.ycol].apply(
                lambda x: value_counts.loc[x]).astype(int)
            self.df[self.ycol] = le.transform(self.df[self.ycol])
            self.df = self.df[self.df['target_count'] > 20]
            self.df = self.df.drop(columns=['target_count'])

            # because n_splits=20 for us
            # self.df = self.df[self.df[self.ycol]]

        assert np.all(np.in1d(list(self.col_action.keys()), self.df.columns))
        return self

    def get_df(self):

        if self.name in self.datasets_with_downloader:
            self.df = getattr(
                datasets.src, self.name
                ).get_df(directory=os.path.split(get_data_path())[0])

        if self.name == 'indultos_espana':
            self.df = pd.read_csv(self.file)

        if self.name == 'open_payments':
            # Variable names in Dollars for Docs dataset ######################
            pi_specialty = ['Physician_Specialty']
            drug_nm = ['Name_of_Associated_Covered_Drug_or_Biological1']
            dev_nm = ['Name_of_Associated_Covered_Device_or_Medical_Supply1']
            corp = ['Applicable_Manufacturer_or_Applicable_GPO_Making_' +
                    'Payment_Name']
            amount = ['Total_Amount_of_Payment_USDollars']
            dispute = ['Dispute_Status_for_Publication']
            ###################################################################

            if os.path.exists(self.file):
                df = pd.read_csv(self.file)
                # print('Loading DataFrame from:\n\t%s' % self.file)
            else:
                csv_files = glob.glob(os.path.join(self.path, 'raw', '*.csv'))
                csv_files_ = []
                for file_ in csv_files:
                    if 'RSRCH_PGYR2013' in file_:
                        csv_files_.append(file_)
                    if 'GNRL_PGYR2013' in file_:
                        csv_files_.append(file_)

                dfd_cols = pi_specialty + drug_nm + dev_nm + corp + amount + \
                    dispute
                df_dfd = pd.DataFrame(columns=dfd_cols)
                for csv_file in csv_files_:
                    if 'RSRCH' in csv_file:
                        df = pd.read_csv(csv_file)
                        df = df[dfd_cols]
                        df['status'] = 'allowed'
                        df = df.drop_duplicates(keep='first')
                        df_dfd = pd.concat([df_dfd, df],
                                           ignore_index=True)
                        print('size: %d, %d' % tuple(df_dfd.shape))
                unique_vals = {}
                for col in df_dfd.columns:
                    unique_vals[col] = set(list(df_dfd[col].unique()))

                for csv_file in csv_files_:
                    if 'GNRL' in csv_file:
                        df = pd.read_csv(csv_file)
                        df = df[dfd_cols]
                        df['status'] = 'disallowed'
                        df = df.drop_duplicates(keep='first')
                        df_dfd = pd.concat([df_dfd, df],
                                           ignore_index=True)
                        print('size: %d, %d' % tuple(df_dfd.shape))
                df_dfd = df_dfd.drop_duplicates(keep='first')
                df_dfd.to_csv(self.file)
                df = df_dfd
            df['status'] = (df['status'] == 'allowed')
            self.df = df
            # print_unique_values(df)
            self.col_action = {
                pi_specialty[0]: 'Delete',
                drug_nm[0]: 'Delete',
                dev_nm[0]: 'Delete',
                corp[0]: 'Special',
                amount[0]: 'Numerical',
                dispute[0]: 'OneHotEncoderDense-1',
                'status': 'y'}
            self.dirtiness_type = {
                corp[0]: 'Synonyms; Overlap'
                }
            self.clf_type = 'binary'

        if self.name == 'road_safety':
            files = self.file
            for filename in files:
                if filename.split('/')[-1] == '2015_Make_Model.csv':
                    df_mod = pd.read_csv(filename, low_memory=False)
                    df_mod['Vehicle_Reference'] = (df_mod['Vehicle_Reference']
                                                   .map(str))
                    df_mod['Vehicle_Index'] = (df_mod['Accident_Index'] +
                                               df_mod['Vehicle_Reference'])
                    df_mod = df_mod.set_index('Vehicle_Index')
                    df_mod = df_mod.dropna(axis=0, how='any', subset=['make'])
            # for filename in files:
            #     if filename.split('/')[-1] == 'Accidents_2015.csv':
            #        df_acc = pd.read_csv(filename).set_index('Accident_Index')
            for filename in files:
                if filename.split('/')[-1] == 'Vehicles_2015.csv':
                    df_veh = pd.read_csv(filename)
                    df_veh['Vehicle_Reference'] = (df_veh['Vehicle_Reference']
                                                   .map(str))
                    df_veh['Vehicle_Index'] = (df_veh['Accident_Index'] +
                                               df_veh['Vehicle_Reference'])
                    df_veh = df_veh.set_index('Vehicle_Index')
            for filename in files:
                if filename.split('/')[-1] == 'Casualties_2015.csv':
                    df_cas = pd.read_csv(filename)
                    df_cas['Vehicle_Reference'] = (df_cas['Vehicle_Reference']
                                                   .map(str))
                    df_cas['Vehicle_Index'] = (df_cas['Accident_Index'] +
                                               df_cas['Vehicle_Reference'])
                    df_cas = df_cas.set_index('Vehicle_Index')

            df = df_cas.join(df_mod, how='left', lsuffix='_cas',
                             rsuffix='_model')
            df = df.dropna(axis=0, how='any', subset=['make'])
            df = df[df['Sex_of_Driver'] != 3]
            df = df[df['Sex_of_Driver'] != -1]
            df['Sex_of_Driver'] = df['Sex_of_Driver'] - 1
            self.df = df
            # col_action = {'Casualty_Severity': 'y',
            #               'Casualty_Class': 'Numerical',
            #               'make': 'OneHotEncoderDense',
            #               'model': 'Special'}
            self.file = self.file[0]

        if self.name == 'consumer_complaints':
            self.df = pd.read_csv(self.file)
            self.df = self.df.dropna(
                axis=0, how='any', subset=['Consumer disputed?'])
            self.df.loc[:, 'Consumer disputed?'] = (
                self.df['Consumer disputed?'] == 'Yes')

        if self.name in ['dating_profiles',
                         'wine_reviews', 'california_housing']:
            self.df = pd.read_csv(self.file)

        if self.name in ['drug_directory']:
            self.df = pd.read_csv(self.file, sep='\t', encoding='latin1')

        if self.name == 'product_relevance':
            self.df = pd.read_csv(self.file, encoding='latin1')

        if self.name == 'federal_election':
            df_dict = pd.read_csv(self.data_dict_file)
            self.df = pd.read_csv(self.file, sep='|', encoding='latin1',
                                  header=None, names=df_dict['Column Name'])
            # Some donations are negative
            self.df['TRANSACTION_AMT'] = self.df['TRANSACTION_AMT'].abs()
            # Predicting the log of the donation
            self.df['TRANSACTION_AMT'] = self.df[
                'TRANSACTION_AMT'].apply(np.log)
            self.df = self.df[self.df['TRANSACTION_AMT'] > 0]

        if self.name == 'cacao_flavors':
            self.df = pd.read_csv(self.file)
            self.df['Cocoa\nPercent'] = self.df[
                'Cocoa\nPercent'].astype(str).str[:-1].astype(float)

        if self.name == 'house_prices':
            self.df = pd.read_csv(self.file, index_col=0)
            # Identifies the type of dwelling involved in the sale.
            MSSubClass = {
                20:	'1-STORY 1946 & NEWER ALL STYLES',
                30:	'1-STORY 1945 & OLDER',
                40:	'1-STORY W/FINISHED ATTIC ALL AGES',
                45:	'1-1/2 STORY - UNFINISHED ALL AGES',
                50:	'1-1/2 STORY FINISHED ALL AGES',
                60:	'2-STORY 1946 & NEWER',
                70:	'2-STORY 1945 & OLDER',
                75:	'2-1/2 STORY ALL AGES',
                80:	'SPLIT OR MULTI-LEVEL',
                85:	'SPLIT FOYER',
                90:	'DUPLEX - ALL STYLES AND AGES',
                120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
                150: '1-1/2 STORY PUD - ALL AGES',
                160: '2-STORY PUD - 1946 & NEWER',
                180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',
                190: '2 FAMILY CONVERSION - ALL STYLES AND AGES',
                }
            for key, value in MSSubClass.items():
                self.df.replace({'MSSubClass': key}, value, inplace=True)

        if self.name == 'kickstarter_projects':
            self.df = pd.read_csv(self.file, encoding='latin1', index_col=0)
            self.df = self.df[self.df['state '].isin(['failed', 'successful'])]
            self.df['state '] = (self.df['state '] == 'successful')
            self.df['usd pledged '] = (
                self.df['usd pledged '].astype(float) + 1E-10).apply(np.log)

        if self.name == 'building_permits':
            self.df = pd.read_csv(self.file)
            self.df.columns = self.df.columns.str.strip()
            self.df['ESTIMATED_COST'] = (
                self.df['ESTIMATED_COST'].astype(float) + 1E-10).apply(np.log)

        if self.name == 'house_sales':
            self.df = pd.read_csv(self.file, index_col=0)

        if self.name == 'vancouver_employee':
            self.df = pd.read_csv(self.file, header=3)
            self.df['Remuneration'] = self.df[
                'Remuneration'].apply(
                    lambda x: np.log(float(''.join(str(x).split(',')))))
        if self.name == 'firefighter_interventions':
            self.df = pd.read_csv(self.file, sep=';')

        if self.name == 'criteo':
            self.df = pd.read_csv(self.file, sep='\t', header=None)
            self.df.columns = self.df.columns.astype(str)

        self.dirty_col = [col for col in self.col_action
                          if self.col_action[col] == 'Special']

        # self.df = self.df[list(self.col_action)]
        return self
