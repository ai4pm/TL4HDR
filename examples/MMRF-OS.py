import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from statsmodels.robust import mad
import numpy as np

from data.preProcess import get_one_race, get_n_years
from data.tcga import read_data
from examples.classify_util import run_mixture_cv, run_one_race_cv, \
    run_unsupervised_transfer_cv


def get_MMRF(X_name, Y_name, groups=('BLACK', 'WHITE'), normalized=True, varthr=0.0, k=400):
    X_path = 'C:/Users/ygao45/Dropbox/TCGA/Other Data/MMRF-COMMPASS/Merged_RNASeq_20190910-114513/' + X_name + '.tsv'
    clinical_path = 'C:/Users/ygao45/Dropbox/TCGA/Other Data/MMRF-COMMPASS/' + Y_name + '.xlsx'
    df_X = pd.read_csv(X_path, sep='\t', index_col='gene_name')
    df_X = df_X.drop(columns=['Unnamed: 0'])
    df_X = df_X.dropna(thresh=df_X.shape[0] * 0.8, axis=1)
    df_X = df_X.dropna()

    patients = list(df_X)
    bar_code = [p[:9] for p in patients]
    df_X.columns = bar_code
    df_X = df_X.T

    df_Y = pd.read_excel(clinical_path, sheet_name='clinical', index_col='submitter_id',
                       usecols='B,G,I,AC') # , 'race', 'days_to_last_follow_up', 'vital_status'

    df_Y.loc[df_Y['race'] == 'asian', 'race'] = 'ASIAN'
    df_Y.loc[df_Y['race'] == 'white', 'race'] = 'WHITE'
    df_Y.loc[df_Y['race'] == 'black or african american', 'race'] = 'BLACK'
    df_Y = df_Y[df_Y['race'].isin(groups)]
    df_Y = df_Y.dropna()

    df_Y = df_Y.rename(columns={"days_to_last_follow_up": "Time"})
    df_Y = df_Y.rename(columns={"vital_status": "C"})
    df_Y = df_Y[df_Y['C'].isin(['Alive', 'Dead'])]
    df_Y = df_Y[df_Y.Time.apply(lambda x: isinstance(x, (int, np.int64)))]
    df_Y.loc[df_Y['C'] == 'Alive', 'C'] = 1
    df_Y.loc[df_Y['C'] == 'Dead', 'C'] = 0

    df = df_X.join(df_Y, how='inner')
    C = df['C'].values.astype('int32')
    E = 1 - C
    R = df['race'].values
    T = df['Time'].values.astype('int32')
    df = df.drop(columns=['C', 'race', 'Time'])

    X = df.values
    X_raw = df.values
    X = X.astype('float32')

    if normalized:
        vt = VarianceThreshold(threshold=varthr)
        vt.fit(X)
        X = vt.transform(X)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    med_dev = pd.DataFrame(mad(X))
    mad_genes = med_dev.sort_values(by=0, ascending=False)\
                       .iloc[0:k].index.tolist()
    X = X[:, mad_genes]
    X_raw = X_raw[:, mad_genes]
    features = list(df.columns[mad_genes])
    Samples = list(df.index)
    print (Samples)
    print (features)
    data = {'X':X, 'X_raw':X_raw, 'T':T, 'C':C, 'E':E, 'R':R, 'Features':features, 'Samples': Samples}

    return data

def run_cv():
    # dataset = get_MMRF(X_name='Merged_FPKM', Y_name='Clinical Data', groups=('WHITE', 'BLACK'), k=600)
    dataset = read_data('MMRF', 'mRNA', 'OS', 3)
    dataset_w = get_one_race(dataset, 'WHITE')
    dataset_w = get_n_years(dataset_w, 3)
    dataset_b = get_one_race(dataset, 'BLACK')
    dataset_b = get_n_years(dataset_b, 3)
    dataset = get_n_years(dataset, 3)
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(y_strat, columns=['RY'])
    df['R'] = R
    df['Y'] = Y
    print(X.shape)
    print(df['RY'].value_counts())
    print(df['R'].value_counts())
    print(df['Y'].value_counts())

    k=-1
    parametrs_b = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':4,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}

    res = pd.DataFrame()
    for i in range(20):
        seed = i
        df_m = run_mixture_cv(seed, dataset, fold=3)
        df_w = run_one_race_cv(seed, dataset_w, fold=3)
        df_w = df_w.rename(columns={"Auc": "W_ind"})
        df_b = run_one_race_cv(seed, dataset_b, **parametrs_b)
        df_b = df_b.rename(columns={"Auc": "B_ind"})
        df_tl = run_unsupervised_transfer_cv(seed, dataset, fold=3)

        df1 = pd.concat([df_m, df_w['W_ind'], df_b['B_ind'], df_tl['TL_Auc']],
                        sort=False, axis=1)
        print(df1)
        res = res.append(df1)
    f_name = 'Result/MM-AA-EA-mRNA-OS-3YR.xlsx'
    res.to_excel(f_name)

def main():
    run_cv()

if __name__ == '__main__':
    main()


