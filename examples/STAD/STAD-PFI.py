from numpy.random import seed
import pandas as pd
import random as rn
import os

from data.preProcess import get_one_race, get_n_years, normalize_dataset, standarize_dataset, get_dataset
from data.tcga import read_data
from examples.classify_util import run_mixture_cv, run_one_race_cv, \
    run_supervised_transfer_cv
from tensorflow import set_random_seed

seed(11111)
set_random_seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)


pd.set_option('display.width', 1000)
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

def run_cv(cancer_type, feature_type, target, years=3, groups=("WHITE", "BLACK")):

    print (cancer_type, feature_type, target, years)
    # dataset = read_data(cancer_type, feature_type[0], target, years)
    dataset = get_dataset(cancer_type=cancer_type, feature_type=feature_type, target=target, groups=("WHITE", "BLACK"))
    dataset = standarize_dataset(dataset)
    dataset_w = get_one_race(dataset, groups[0])
    dataset_w = get_n_years(dataset_w, years)
    dataset_b = get_one_race(dataset, groups[1])
    dataset_b = get_n_years(dataset_b, years)
    dataset = get_n_years(dataset, years)

    k = 200 if 'mRNA' in feature_type else -1
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(y_strat, columns=['RY'])
    df['R'] = R
    df['Y'] = Y
    Dict = df['RY'].value_counts()
    Dict = dict(Dict)
    print (Dict)

    parametrs_mix = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,'momentum':0.9,
                     'learning_rate':0.01, 'lr_decay':0.03, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_w = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}

    parametrs_b = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':4,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}

    parametrs_tl = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':32, 'tune_epoch':100, 'train_epoch':100,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5, 'tune_lr':0.01,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64], 'tune_batch':10}

    res = pd.DataFrame()
    for i in range(20):
        seed = i
        df_m = run_mixture_cv(seed, dataset, **parametrs_mix, groups=groups)
        df_w = run_one_race_cv(seed, dataset_w, **parametrs_w)
        df_w = df_w.rename(columns={"Auc": "W_ind"})
        df_b = run_one_race_cv(seed, dataset_b, **parametrs_b)
        df_b = df_b.rename(columns={"Auc": "A_ind"})
        df_tl = run_supervised_transfer_cv(seed, dataset, **parametrs_tl, groups=groups)
        df1 = pd.concat([df_m, df_w['W_ind'], df_b['A_ind'], df_tl['TL_Auc']],
                        sort=False, axis=1)

        print (df1)
        res = res.append(df1)

    f_name = 'Result/' + cancer_type + '-AA-EA-' + feature_type[0] + '-' + target + '-' + str(years) + 'YR.xlsx'
    res.to_excel(f_name)


def main():

    run_cv('STAD', 'Protein', 'DSS', years=1, groups=("WHITE", "ASIAN"))

if __name__ == '__main__':
    main()
