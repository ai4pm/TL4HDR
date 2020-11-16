from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import os

from pathlib import Path
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import scipy.io as sio

home_path = os.path.dirname(__file__) + '/datasets/'

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

def get_protein(cancer_type, target='OS', groups=("WHITE", "BLACK")):
    path = home_path + 'Protein.txt'
    df = pd.read_csv(path, sep='\t', index_col='SampleID')
    df = df.dropna(axis=1)
    tumorTypes = tumor_types(cancer_type)
    df = df[df['TumorType'].isin(tumorTypes)]
    df = df.drop(columns=['TumorType'])
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new

    return add_race_CT(cancer_type, df, target, groups)

def get_mRNA(cancer_type, target='OS', groups=("WHITE", "BLACK")):
    path = home_path + 'mRNA.mat'
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

    # Packing the data
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

def normalize_dataset(data):
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
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

def get_CT(target):
    path1 = home_path + 'TCGA-CDR-SupplementalTableS1.xlsx'
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
    return df_C_T

def get_race(cancer_type):
    path = home_path + 'Genetic_Ancestry.xlsx'
    df_list = [pd.read_excel(path, disease, usecols='A,E', index_col='Patient_ID', keep_default_na=False)
               for disease in tumor_types(cancer_type)]
    df_race = pd.concat(df_list)
    df_race = df_race[df_race['EIGENSTRAT'].isin(['EA', 'AA', 'EAA', 'NA', 'OA'])]
    df_race['race'] = df_race['EIGENSTRAT']

    df_race.loc[df_race['EIGENSTRAT'] == 'EA', 'race'] = 'WHITE'
    df_race.loc[df_race['EIGENSTRAT'] == 'AA', 'race'] = 'BLACK'
    df_race.loc[df_race['EIGENSTRAT'] == 'EAA', 'race'] = 'ASIAN'
    df_race.loc[df_race['EIGENSTRAT'] == 'NA', 'race'] = 'NAT_A'
    df_race.loc[df_race['EIGENSTRAT'] == 'OA', 'race'] = 'OTHER'
    df_race = df_race.drop(columns=['EIGENSTRAT'])

    return df_race

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
