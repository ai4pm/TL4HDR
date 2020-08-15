
import pandas as pd
from scipy.io import savemat, loadmat
import numpy as np
import os

pd.set_option('display.width', 1000)
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

# home_path = 'C:/Users/ygao45/Documents/GitHub/TL4RRD/data/datasets/'
home_path = os.path.dirname(__file__) + '/datasets/'
def read_data(cancer_type, feature_type, target, years):
    # home_path = 'E:/PythonWorkSpace/TransferLearning-upload/data/datasets/'
    path = home_path + cancer_type + '-AA-EA-' + feature_type + '-' + target + '-' + str(years) + 'YR.mat'
    dataset = loadmat(path)

    X, T, C, E, R = dataset['X'], dataset['T'][0], dataset['C'][0], dataset['E'][0], dataset['R'][0]
    data  = {'X': X, 'T': T, 'C': C, 'E': E, 'R': R}

    return data

def get_one_race(dataset, race):
    X, T, C, E, R = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R']
    mask = R == race
    X, T, C, E, R = X[mask], T[mask], C[mask], E[mask], R[mask]
    data = {'X': X, 'T': T, 'C': C, 'E': E, 'R': R}
    return data

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
	


dataset = read_data('MMRF', 'mRNA', 'OS', 3)


