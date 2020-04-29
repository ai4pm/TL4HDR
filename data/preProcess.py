from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import os

from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import scipy.io as sio

home_path = 'C:/Users/ygao45/Dropbox/'

def get_input_path(cancer_type):
    return home_path + 'TCGA/Progress/dataset/Other Cancers/' + cancer_type + '.Data.mat'

def get_output_path(cancer_type, feature_type):
    folder_path = 'E:/SurvivalAnalysis/' + cancer_type
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = 'E:/SurvivalAnalysis/' + cancer_type + '/' + cancer_type + '_' + feature_type + '.mat'
    return path

def get_clinical_follow_up_path():
    path = home_path + '/TCGA/Data/Clinical/clinical_PANCAN_patient_with_followup.tsv'
    return path


def tumor_types(cancer_type):
    Map = {'GBMLGG': ['GBM', 'LGG'],
           'COADREAD': ['COAD', 'READ'],
           'KIPAN': ['KIRC', 'KICH', 'KIRP'],
           'STES': ['ESCA', 'STAD'],
           'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'],
           'PanGyn': ['OV', 'CESC', 'USC', 'UCEC'],
           'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'],
           'PanPan': ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
                           'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
                           'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ',
                           'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
           }
    if cancer_type not in Map:
        Map[cancer_type] = [cancer_type]

    return Map[cancer_type]

def get_data_microRNA(cancer_type, target='OS', groups=('BLACK', 'WHITE')):
    X_path = home_path + '/TCGA/Data/microRNA/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv'
    Y_path = home_path + '/TCGA/Data/microRNA/PanCanAtlas_miRNA_sample_information_list.txt'
    df_X = pd.read_csv(X_path, index_col='Genes')
    df_X = df_X[df_X['Correction'] == 'Corrected']
    df_X = df_X.drop(columns=['Correction'])

    df_Y = pd.read_csv(Y_path, sep='\t')
    df_Y = df_Y[df_Y['Disease'].isin(tumor_types(cancer_type))]
    patients = df_Y['id'].values
    bar_code = [p[:12] for p in patients]
    df_X = df_X[patients]
    df_X.columns = bar_code
    df_X = df_X.T

    data = add_race_CT(cancer_type, df_X, target, groups)
    return data

def get_protein(cancer_type, target='OS', groups=("WHITE", "BLACK")):
    path = home_path + '/TCGA/Data/Protein/Protein.txt'
    df = pd.read_csv(path, sep='\t', index_col='SampleID')
    df = df.dropna(axis=1)
    tumorTypes = tumor_types(cancer_type)
    df = df[df['TumorType'].isin(tumorTypes)]
    df = df.drop(columns=['TumorType'])
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new

    return add_race_CT(cancer_type, df, target, groups)

def get_methylation(cancer_type, target='OS', groups=("WHITE", "BLACK")):
    path = home_path + '/TCGA/Data/Methylation/PanCanerMeth_ClinInfo.mat'

    A = loadmat(path)
    X, Y, GeneName, SampleName = A['X'].astype('float32'), A['Y'], A['GeneName'][0], A['SampleName']
    Y = [row[0] for row in Y]
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]

    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    tumorTypes = tumor_types(cancer_type)
    df_Y = df_Y[df_Y['Disease'].isin(tumorTypes)]

    df = df_X.join(df_Y, how='inner')
    df = df.drop(columns=['Disease'])
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    return add_race_CT(cancer_type, df, target, groups)

def get_clinical(cancer_type, target='OS', groups=("WHITE", "BLACK")):
    path = home_path + '/TCGA/Data/Clinical/Common-Used-Clinical-Features.xlsx'

    df = pd.read_excel(path, sheet_name='Clinical')
    df = df[df['acronym'].isin([tumor_types(cancer_type)])]
    df = df.set_index('bcr_patient_barcode')
    df = df.drop(columns=['acronym'])
    df = df.dropna(axis='columns')

    # filter each column
    df = df[df['gender'].isin(['MALE', 'FEMALE'])]
    df = df[~df['age_at_initial_pathologic_diagnosis'].isin(['[Not Available]'])]
    df = df[~df['histological_type'].isin(['[Not Available]'])]

    df = df[~df['pathologic_stage'].isin(["", '[Unknown]', '[Discrepancy]', '[Not Available]',
                                          '[Not Applicable]'])]
    df = df[~df['pathologic_M'].isin(["", '[Unknown]', '[Discrepancy]', '[Not Available]',
                                      '[Not Applicable]'])]

    df = df[~df['pathologic_N'].isin(["", '[Unknown]', '[Discrepancy]', '[Not Available]',
                                      '[Not Applicable]'])]
    df = df[~df['pathologic_T'].isin(["", '[Unknown]', '[Discrepancy]', '[Not Available]',
                                      '[Not Applicable]'])]
    df = df[~df['radiation_therapy'].isin(["", '[Unknown]', '[Discrepancy]', '[Not Available]',
                                           '[Not Applicable]'])]

    df1 = df[['age_at_initial_pathologic_diagnosis']]
    df2 = df.drop(columns=['age_at_initial_pathologic_diagnosis'])
    df2 = pd.get_dummies(df2)
    drop_cols = df2.columns[df2.sum(axis=0, skipna=True) < 0.1 * df2.shape[0]]
    df2.drop(drop_cols, axis=1, inplace=True)

    df = df1.join(df2, how='inner')
    return add_race_CT(cancer_type, df, target, groups)

def get_mRNA(cancer_type, target='OS', groups=("WHITE", "BLACK")):
    path = home_path + '/TCGA/Data/Transcriptome/PanCanerRNA.mat'
    A = loadmat(path)
    X, Y, GeneName, SampleName = A['X'].astype('float32'), A['Y'], A['GeneName'][0], A['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]

    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    df_Y = df_Y[df_Y['Disease'].isin(tumor_types(cancer_type))]
    df = df_X.join(df_Y, how='inner')
    df = df.drop(columns=['Disease'])

    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    return add_race_CT(cancer_type, df, target, groups)


def add_race_CT(cancer_type, df, target, groups):
    df_race = get_race(cancer_type)
    df_race = df_race[df_race['race'].isin(groups)]
    df_C_T = get_CT(target)

    # Keep patients with race information
    df = df.join(df_race, how='inner')
    print(df.shape)
    df = df.dropna(axis='columns')
    df = df.join(df_C_T, how='inner')
    print(df.shape)

    # Packing the data and save it to the disk
    C = df['C'].tolist()
    R = df['race'].tolist()
    T = df['T'].tolist()
    E = [1 - c for c in C]
    df = df.drop(columns=['C', 'race', 'T'])
    X = df.values
    X = X.astype('float32')
    data = {'X': X, 'T': np.asarray(T, dtype=np.float32),
            'C': np.asarray(C, dtype=np.int32), 'E': np.asarray(E, dtype=np.int32),
            'R': np.asarray(R), 'Samples': df.index.values, 'FeatureName': list(df)}

    return data

def get_fn(feature_type):
    fn = get_protein
    if feature_type == 'mRNA':
        fn = get_mRNA
    elif feature_type == 'methylation':
        fn = get_methylation
    elif feature_type == 'microRNA':
        fn = get_data_microRNA
    return fn

def get_dataset(cancer_type, feature_type='Integ', target='OS', groups=("WHITE", "BLACK")):
    fn = get_fn(feature_type)
    return fn(cancer_type, target=target, groups=groups)


def get_n_years(dataset, years):
    X, T, C, E, R = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R']

    df = pd.DataFrame(X)
    df['T'] = T
    df['C'] = C
    df['R'] = R
    df['Y'] = 1

    df = df[~((df['T'] < 365 * years) & (df['C'] == 1))]
    df.loc[df['T'] <= 365 * years, 'Y'] = 0
    df['strat'] = df.apply(lambda row: str(row['Y']) + str(row['R']), axis=1)
    df = df.reset_index(drop=True)

    R = df['R'].values
    Y = df['Y'].values
    y_strat = df['strat'].values
    df = df.drop(columns=['T', 'C', 'R', 'Y', 'strat'])
    X = df.values
    y_sub = R # doese not matter

    return (X, Y.astype('int32'), R, y_sub, y_strat)

def get_dataset_integ(cancer_type, feature_type=['Protein'], target='OS', groups=("WHITE", "BLACK")):
    datasets = []
    for feature in feature_type:
        fn = get_fn(feature)
        data = fn(cancer_type, target=target, groups=groups)
        datasets.append(data)

    return merge_datasets(datasets)


def merge_datasets(datasets):

    data = datasets[0]
    X, T, C, E, R, Samples, FeatureName = data['X'], data['T'], data['C'], data['E'], data['R'], data['Samples'], data['FeatureName']
    df = pd.DataFrame(X, index=Samples, columns=FeatureName)
    df['T'] = T
    df['C'] = C
    df['E'] = E
    df['R'] = R

    for i in range(1, len(datasets)):
        data1 = datasets[i]
        X1, Samples, FeatureName = data1['X'], data1['Samples'], data1['FeatureName']
        temp = pd.DataFrame(X1, index=Samples, columns=FeatureName)
        df = df.join(temp, how='inner')

    # Packing the data and save it to the disk
    C = df['C'].tolist()
    R = df['R'].tolist()
    T = df['T'].tolist()
    E = df['E'].tolist()
    df = df.drop(columns=['C', 'R', 'T', 'E'])
    X = df.values
    X = X.astype('float32')
    data = {'X': X, 'T': np.asarray(T, dtype=np.float32),
            'C': np.asarray(C, dtype=np.int32), 'E': np.asarray(E, dtype=np.int32),
            'R': np.asarray(R), 'Samples': df.index.values, 'FeatureName': list(df)}

    return data


def get_dataset_by_gender(cancer_type, target='OS', gender=("MALE", "FEMALE")):
    path = home_path + '/TCGA/Data/Protein/Protein.txt'
    df = pd.read_csv(path, sep='\t', index_col='SampleID')
    df = df.dropna(axis=1)
    tumorTypes = tumor_types(cancer_type)
    df = df[df['TumorType'].isin(tumorTypes)]
    df = df.drop(columns=['TumorType'])
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new

    data = add_gender_CT(cancer_type, df, target, gender)
    return data

def add_gender_CT(cancer_type, df, target, gender):
    df_C_T = get_CT(target)
    df_gender = get_gender(tumor_types(cancer_type))
    df_gender = df_gender[df_gender['gender'].isin(gender)]

    # Keep patients with race information
    df = df.join(df_gender, how='inner')
    print(df.shape)
    df = df.dropna(axis='columns')
    df = df.join(df_C_T, how='inner')
    print(df.shape)

    # Packing the data and save it to the disk
    C = df['C'].tolist()
    R = df['gender'].tolist()
    T = df['T'].tolist()
    E = [1 - c for c in C]
    df = df.drop(columns=['C', 'gender', 'T'])
    X = df.values
    X = X.astype('float32')
    data = {'X': X, 'T': np.asarray(T, dtype=np.float32),
            'C': np.asarray(C, dtype=np.int32), 'E': np.asarray(E, dtype=np.int32),
            'R': np.asarray(R), 'Samples': df.index.values, 'FeatureName': list(df)}

    return data


# get data downloaded by TCGAIntegrator
def get_data(cancer_type, feature_type='Integ', target='OS', groups=('BLACK', 'WHITE'), normalized=True):
    # dataset_path = 'E:/SurvivalAnalysis/' + cancer_type + '/' + cancer_type + '_' + feature_type + '.mat'
    # dataset_path = 'C:/Users/gaoy/Documents/Dropbox/temp/Data/' + cancer_type + '/' + target + '/' + feature_type + '.mat'

    dataset_path = home_path + '/temp/Data/Data/' + cancer_type + '/' + target + '/' + feature_type + '.mat'
    A = loadmat(dataset_path)
    X, T, C, R = A['X'].astype('float32'), A['T'][0].astype('float32'), A['C'][0].astype('int32'), A['R']

    for i in range(len(R)):
        R[i] = R[i][0:5]
    index = [idx for idx, val in enumerate(R) if val in groups]
    X, T, C, R = X[index], T[index], C[index], R[index]

    if normalized:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    return X, T, C, R


def normalize_dataset(data, thr=0.00):
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    if thr > 0:
        vt = VarianceThreshold(threshold=thr)
        vt.fit(X)
        X = vt.transform(X)
    X = preprocessing.normalize(X)
    data_new['X'] = X
    return data_new


def standarize_dataset(data):
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    data_new['X'] = X
    return data_new



def get_one_race(dataset, race):
    X, T, C, E, R = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R']
    mask = R == race
    X, T, C, E, R = X[mask], T[mask], C[mask], E[mask], R[mask]
    data = {'X': X, 'T': T, 'C': C, 'E': E, 'R': R}
    return data

def get_one_race_clf(dataset, race):
    X, Y, R, y_sub, y_strat = dataset
    mask = R == race
    X, Y, R, y_sub, y_strat = X[mask], Y[mask], R[mask], y_sub[mask], y_strat[mask]
    return (X, Y, R, y_sub, y_strat)

def get_eachType(featureType, target):
    df = pd.DataFrame()
    for cancer_type in ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
                        'ESCA', 'GBM', 'HNSC', 'KIRC', 'KIRP', 'LGG',
                        'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PCPG', 'PRAD',
                        'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM',
                        'UCEC', 'UCS', 'GBMLGG', 'COADREAD', 'KIPAN', 'STES']:
        # LAML, KICH, UVM, 'MESO' are missing
        # 'GBMLGG', 'COADREAD', 'KIPAN', 'STES',  can be added later
        df1 = get_data_microRNA(cancer_type, featureType, target)
        df = df.append(df1)
    return df



def get_ras_data(race='', cancer_type='Pan', norm=True):
    path1 = home_path + '/temp/Ras_pathway/Data/White_Black.mat'
    data = loadmat(path1)
    Features = data['Features']
    Samples = data['Samples']
    X = data['X'].astype('float32')
    Y = list(data['Y'][0].astype('int32'))
    R = [row[0] for row in data['race'][0]]
    y_sub = [row[0] for row in data['y_sub'][0]]

    if race != '':
        mask = [row == race for row in R]
        X = X[mask]
        R = [R[idx] for idx, val in enumerate(mask) if val]
        Y = [Y[idx] for idx, val in enumerate(mask) if val]
        y_sub = [y_sub[idx] for idx, val in enumerate(mask) if val]
        Samples = [Samples[idx] for idx, val in enumerate(mask) if val]

    if cancer_type != 'Pan':
        mask = [row == cancer_type for row in y_sub]
        X = X[mask]
        R = [R[idx] for idx, val in enumerate(mask) if val]
        Y = [Y[idx] for idx, val in enumerate(mask) if val]
        y_sub = [y_sub[idx] for idx, val in enumerate(mask) if val]
        Samples = [Samples[idx] for idx, val in enumerate(mask) if val]

    if norm:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    y_strat = [str(R[i]) + str(Y[i]) for i, r in enumerate(R)]
    X = pd.DataFrame(X, columns=Features, index=Samples)
    Y = pd.DataFrame(Y, index=Samples, columns=['Y'])
    R = pd.DataFrame(R, index=Samples, columns=['R'])
    y_sub = pd.DataFrame(y_sub, index=Samples, columns=['y_sub'])
    y_strat = pd.DataFrame(y_strat, index=Samples, columns=['y_strat'])

    return X.values, Y['Y'].values.astype('int32'), R['R'].values, y_sub['y_sub'].values, y_strat['y_strat'].values


# deprecated interfaces
# transform the raw data into a dictionary:
# X: n*p array n samples, p variables, raw values
# R: Race of each patient
# C: The censored status of each patient
# T: The survival time or the last follow-up time of each patient.
# E: The event indicator of each patient
def transform_raw_data(cancer_type, feature_type, target='OS', df_race=None, df_C_T=None, thr=0.8):
    path = get_input_path(cancer_type)
    A = loadmat(path)
    Features = A['Features']
    Samples = A['Samples']
    Samples = [sample[0:12] for sample in Samples]
    Symbols = A['Symbols']
    SymbolTypes = A['SymbolTypes']
    Censored = A['Censored'][0]
    Survival = A['Survival'][0]

    # pre-select features
    use_types = []
    if feature_type == 'Integ':
        use_types = [u'CNVArm  ', u'CNVGene ', u'Clinical', u'Mutation', u'Protein ']
    elif feature_type == 'Omics':
        use_types = ['mRNA    ']
    elif feature_type == 'Integ_omics':
        use_types = [u'CNVArm  ', u'CNVGene ', u'Clinical', u'Mutation', u'Protein ', 'mRNA    ']
    elif feature_type == 'Protein':
        use_types = [u'Protein ']

    A_Indices = [Index for Index, Type in enumerate(SymbolTypes) if Type in use_types]
    A_Selected = Features[A_Indices, :]
    A_Selected = A_Selected.transpose()
    A_featureName = [Symbols[i] for i in A_Indices]
    df = pd.DataFrame(A_Selected, columns=A_featureName, index=Samples)
    n = df.shape[1]
    print(cancer_type, feature_type, df.shape, ' ->  ')

    if df_C_T:
        df_C_T = pd.DataFrame()
        df_C_T['C'] = pd.Series(Censored, index=Samples)
        df_C_T['T'] = pd.Series(Survival, index=Samples)
        df_C_T.dropna(inplace=True)
    else:
        path1 = 'C:/Users/gaoy/Documents/Dropbox/TCGA/Data/Clinical/TCGA-CDR-SupplementalTableS1.xlsx'
        cols = 'B,Z,AA'
        if target == 'DSS':
            cols = 'B,AB,AC'
        elif target == 'DFI':
            cols = 'B,AD,AE'
        elif target == 'PFI':
            cols = 'B,AF,AG'

        df_C_T = pd.read_excel(path1, 'TCGA-CDR', usecols=cols, index_col='bcr_patient_barcode')
        df_C_T.columns = ['E', 'T']
        df_C_T = df_C_T[df_C_T['E'].isin([0, 1])]
        df_C_T = df_C_T.dropna()
        df_C_T['C'] = 1 - df_C_T['E']
        df_C_T.drop(columns=['E'], inplace=True)

    # remove clinical race features
    if feature_type in ['Integ', 'Integ_omics']:
        cols = [c for c in df.columns if c[0:8] == 'race-Is-']
        df.drop(columns=cols, inplace=True)

    # if no race information, get the race information from clinical follow up file, remove nan values
    if df_race is None:
        df_race = pd.read_csv(get_clinical_follow_up_path(), sep='\t', index_col='bcr_patient_barcode',
                              usecols=['bcr_patient_barcode', 'race'])
        df_race.dropna(inplace=True)
        # race in  one of ['AMERICAN INDIAN OR ALASKA NATIVE', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
        # 'WHITE', 'BLACK OR AFRICAN AMERICAN', 'ASIAN', '[Not Evaluated]', '[Unknown]', '[Not Available]']
        # df_race = df_race[df_race['race'].isin(['WHITE', 'BLACK OR AFRICAN AMERICAN', 'ASIAN'])]
        df_race = df_race[~df_race['race'].isin(['[Not Evaluated]', '[Unknown]', '[Not Available]'])]
        mask = df_race['race'] == 'BLACK OR AFRICAN AMERICAN'
        df_race.loc[mask, 'race'] = 'BLACK'

    # Keep patients with race information
    df = df.join(df_race, how='inner')

    # Drop patients with >= 20% missing values, then remove features with missing values.
    df = df.dropna(thresh=int(thr * n))
    df = df.dropna(axis='columns')
    df = df.join(df_C_T, how='inner')

    # Packing the data and save it to the disk
    C = df['C'].tolist()
    R = df['race'].tolist()
    T = df['T'].tolist()
    E = [1 - c for c in C]
    df = df.drop(columns=['C', 'race', 'T'])
    X = df.values
    data = {'X': X, 'R': R, 'C': C, 'T': T, 'E': E, 'Samples': df.index.values, 'FeatureName': list(df)}
    print(df.shape)
    out_file_name = get_output_path(cancer_type, feature_type)
    savemat(out_file_name, data)
    print(' White: ', R.count('WHITE'), 'Black: ', R.count('BLACK'))
    return data

def transform_brain_integ():
    path = r'E:\PythonWorkSpace\SurvivalNet-master\data\Brain_Integ.mat'
    A = loadmat(path)
    Features = A['Integ_Symbs']
    Samples = A['Patients']
    Samples = [sample[0:12] for sample in Samples]
    # Symbols = A['Symbols']
    Censored = np.squeeze(A['Censored'])
    Survival = np.squeeze(A['Survival'])
    X = A['Integ_X']

    print(Censored)
    print(Survival)

    df_X = pd.DataFrame(X, columns=Features, index=Samples)

    # if feature_type in ['Integ', 'Integ_omics'], drop race features:
    cols = [c for c in df_X.columns if c[0:8] == 'race-Is-']
    df_X.drop(columns=cols, inplace=True)

    df_race = pd.read_csv(get_clinical_follow_up_path(), sep='\t', index_col='bcr_patient_barcode',
                          usecols=['bcr_patient_barcode', 'race'])
    df_race.dropna(inplace=True)
    # race in  one of ['AMERICAN INDIAN OR ALASKA NATIVE', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
    # 'WHITE', 'BLACK OR AFRICAN AMERICAN', 'ASIAN', '[Not Evaluated]', '[Unknown]', '[Not Available]']
    # df_race = df_race[df_race['race'].isin(['WHITE', 'BLACK OR AFRICAN AMERICAN', 'ASIAN'])]
    df_race = df_race[~df_race['race'].isin(['[Not Evaluated]', '[Unknown]', '[Not Available]'])]
    mask = df_race['race'] == 'BLACK OR AFRICAN AMERICAN'
    df_race.loc[mask, 'race'] = 'BLACK'

    df_C_T = pd.DataFrame()
    df_C_T['Censored'] = pd.Series(Censored, index=Samples)
    df_C_T['Survival'] = pd.Series(Survival, index=Samples)
    df_C_T.dropna(inplace=True)

    df_C_T_R = df_C_T.join(df_race, how='inner')

    # Packing the data and save it to the disk
    C = df_C_T_R['Censored'].tolist()
    R = df_C_T_R['race'].tolist()
    T = df_C_T_R['Survival'].tolist()
    E = [1 - c for c in C]
    df_X = df_X.loc[df_C_T_R.index, :]
    X = df_X.values
    data = {'X': X, 'R': R, 'C': C, 'T': T, 'E': E, 'Samples': df_X.index.values, 'FeatureName': list(df_X)}
    print(df_X.shape)
    out_file_name = get_output_path('Brain', 'Integ')
    savemat(out_file_name, data)
    print(' White: ', R.count('WHITE'), 'Black: ', R.count('BLACK'))

def tranform_PRAD(cancer_type, feature_type):
    path = home_path + '/TCGA/Data/Genetic_Ancestry_PRAD.xlsx'
    df = pd.read_excel(path, 'Sheet2', usecols='A,E', index_col='Patient_ID')
    print(df.shape, '->')
    df = df[df['EIGENSTRAT'].isin(['EA', 'AA'])]
    df['race'] = np.where(df['EIGENSTRAT'] == 'EA', 'WHITE', 'BLACK')
    df_race = df['race']
    print(df_race.shape)
    print(df_race.value_counts())

    path1 = 'C:/Users/gaoy/Documents/Dropbox/TCGA/Data/Clinical/TCGA-CDR-SupplementalTableS1.xlsx'
    df_C_T = pd.read_excel(path1, 'Sheet1', usecols='B,AF,AG', index_col='bcr_patient_barcode')
    print(df_C_T.shape)
    df_C_T['C'] = 1 - df_C_T['PFI']
    df_C_T = df_C_T.rename(columns={"PFI.time": "T"})
    df_C_T.drop(columns=['PFI'], inplace=True)

    data = transform_raw_data(cancer_type, feature_type, df_race=df_race, df_C_T=df_C_T, thr=0.8)
    out_file_name = get_output_path(cancer_type + '_corrected_PFI', feature_type)
    savemat(out_file_name, data)

def correct_CORE():
    path = home_path + '/TCGA/Data/Protein/Protein.txt'
    df = pd.read_csv(path, sep='\t')
    df = df.dropna(axis=1)
    df = df[df['TumorType'] == 'CORE']
    df = df[['TumorType', 'SampleID', 'X1433EPSILON']]
    print(df)

    Y_path = home_path + '/TCGA/Data/Protein/merged_sample_quality_annotations.xlsx'
    df_Y = pd.read_excel(Y_path)
    df_Y = df_Y.rename(columns={"aliquot_barcode": "SampleID"})
    # df_Y['aliquot_barcode'] = df['aliquot_barcode'].apply(lambda x: x.replace('_', '-'))
    # df_Y = df_Y.set_index('aliquot_barcode')
    index = df_Y.SampleID.values
    index_new = [row.replace('_', '-') for row in index]
    df_Y['SampleID'] = index_new
    df_Y = df_Y[df_Y['cancer type'].isin(['COAD', 'READ'])]
    df_Y = df_Y[['cancer type', 'SampleID', 'patient_barcode']]

    print(df_Y.shape)
    df2 = df.set_index('SampleID').join(df_Y.set_index('SampleID'), how='left', on='SampleID')
    print(df2)

def main():

    path = home_path + '/TCGA/Data/Protein/Protein.txt'
    df = pd.read_csv(path, sep='\t')
    df = df.dropna(axis=1)
    df = df[df['TumorType']=='CORE']
    df = df[['TumorType', 'SampleID', 'X1433EPSILON']]
    print (df)

    Y_path = home_path + '/TCGA/Data/Protein/merged_sample_quality_annotations.xlsx'
    df_Y = pd.read_excel(Y_path)
    df_Y = df_Y.rename(columns={"aliquot_barcode": "SampleID"})
    # df_Y['aliquot_barcode'] = df['aliquot_barcode'].apply(lambda x: x.replace('_', '-'))
    # df_Y = df_Y.set_index('aliquot_barcode')
    index = df_Y.SampleID.values
    index_new = [row.replace('_', '-') for row in index]
    df_Y['SampleID'] = index_new
    df_Y = df_Y[df_Y['cancer type'].isin(['COAD', 'READ'])]
    df_Y = df_Y[['cancer type', 'SampleID', 'patient_barcode']]

    print (df_Y.shape)
    df2 = df.set_index('SampleID').join(df_Y.set_index('SampleID'), how='left', on='SampleID')
    print(df2)
	


if __name__ == '__main__':
    main()
