from numpy.random import seed
import pandas as pd
import random as rn
import os
from sklearn import preprocessing

from data.preProcess import get_one_race_clf
from examples.classify_util import run_mixture_cv, run_one_race_cv, \
    run_supervised_transfer_cv
from tensorflow import set_random_seed

from simulation.simulate_data_clf import SimulatedData

seed(11111)
set_random_seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)

def run_cv():
    factory = SimulatedData(
        num_var=200,
        pos = [0, 0, 0, 0],
        neg=[0, 0, 0, 0],
        neut=[100,0,0,100]
    )

    dataset = factory.generate_black_white_data('PanGyn-DFI-5-base.mat',
                                                white_alive=130, white_dead=130, black_alive=130, black_dead=130)
    dataset_w = get_one_race_clf(dataset, 'WHITE')
    dataset_b = get_one_race_clf(dataset, 'BLACK')
    dataset_tl = [e for e in dataset]
    dataset_tl[0] = preprocessing.normalize(dataset_tl[0])

    k = -1
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(y_strat, columns=['RY'])
    df['R'] = R
    df['Y'] = Y
    print(X.shape)
    print(df['RY'].value_counts())
    print(df['R'].value_counts())
    print(df['Y'].value_counts())

    parametrs_mix = {'fold': 3, 'k': k, 'val_size': 0.0, 'batch_size': 20, 'momentum': 0.9,
                     'learning_rate': 0.01, 'lr_decay': 0.0, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_w = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}

    parametrs_b = {'fold': 3, 'k': k, 'val_size': 0.0, 'batch_size': 20,
                   'learning_rate': 0.01, 'lr_decay': 0.0, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}

    parametrs_tl = {'fold': 3, 'k': k, 'val_size': 0.0, 'batch_size': 32, 'train_epoch': 100, 'tune_epoch': 100,
                    'learning_rate': 0.01, 'lr_decay': 0.0, 'dropout': 0.5, 'tune_lr': 0.001,
                    'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64], 'tune_batch': 32}

    res = pd.DataFrame()
    for i in range(20):
        seed = i
        df_m = run_mixture_cv(seed, dataset, **parametrs_mix)
        df_w = run_one_race_cv(seed, dataset_w, **parametrs_w)
        df_w = df_w.rename(columns={"Auc": "W_ind"})
        df_b = run_one_race_cv(seed, dataset_b, **parametrs_b)
        df_b = df_b.rename(columns={"Auc": "B_ind"})
        df_tl = run_supervised_transfer_cv(seed, dataset, **parametrs_tl)
        df1 = pd.concat([df_m, df_w['W_ind'], df_b['B_ind'], df_tl['TL_Auc']],
                        sort=False, axis=1)
        print (df1)
        res = res.append(df1)

    print (res)


def main():
    run_cv()

if __name__ == '__main__':
    main()
