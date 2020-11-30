from numpy.random import seed
import pandas as pd
import random as rn
import os, sys

from data.preProcess import get_one_race, get_n_years, normalize_dataset, get_dataset_integ, \
    standarize_dataset, get_dataset
from examples.classify_util import run_mixture_cv, run_one_race_cv, \
    run_unsupervised_transfer_cv, run_supervised_transfer_cv, run_CCSA_transfer
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
    # dataset = get_dataset_integ(cancer_type=cancer_type, feature_type=feature_type, target=target, groups=groups)
    dataset = get_dataset(cancer_type=cancer_type, feature_type=feature_type, target=target, groups=groups)
    if dataset['X'].shape[0] < 10: return None
    dataset = standarize_dataset(dataset)
    dataset_w = get_one_race(dataset, 'WHITE')
    if dataset_w['X'].shape[0] < 5: return None
    dataset_w = get_n_years(dataset_w, years)
    dataset_b = get_one_race(dataset, 'BLACK')
    if dataset_b['X'].shape[0] < 5: return None
    dataset_b = get_n_years(dataset_b, years)

    dataset_tl = normalize_dataset(dataset)
    dataset_tl = get_n_years(dataset_tl, years)

    dataset = get_n_years(dataset, years)
    k = 200 if 'mRNA' in feature_type or 'methylation' in feature_type else -1

    # print(numpy.count_nonzero(numpy.isnan(dataset['X'])))
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(y_strat, columns=['RY'])
    df['R'] = R
    df['Y'] = Y
    print(X.shape)
    Dict = df['RY'].value_counts()
    print (Dict)
    if len(Dict) < 4: return None
    Dict = dict(Dict)
    print (Dict)
    for key in Dict:
        print (key, Dict[key])
        if Dict[key] < 5:
            return None

    parametrs_mix = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,'momentum':0.9,
                     'learning_rate':0.01, 'lr_decay':0.03, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_w = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_b = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':4,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}

    parametrs_tl = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20, 'tune_epoch':500,
                     'learning_rate':0.01, 'lr_decay':0.03, 'dropout':0.5, 'tune_lr':0.002,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64], 'tune_batch':10}

    parametrs_tl_unsupervised = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
                     'learning_rate':0.001, 'lr_decay':0.03, 'dropout':0.0, 'n_epochs':100,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}

    # parametrs_tl_sa = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
    #                  'learning_rate':0.005, 'lr_decay':0.0, 'dropout':0.5,
    #                  'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}


    parameters_CCSA = {'fold': 3, 'n_features': k, 'alpha':0.3, 'batch_size':32, 'learning_rate':0.01,
                       'hiddenLayers': [100], 'dr':0.0, 'momentum':0.0,
                       'decay':0.0, 'sample_per_class':2}

    res = pd.DataFrame()
    for i in range(20):
        seed = i
        df_m = run_mixture_cv(seed, dataset, **parametrs_mix)
        df_w = run_one_race_cv(seed, dataset_w, **parametrs_w)
        df_w = df_w.rename(columns={"Auc": "W_ind"})
        df_b = run_one_race_cv(seed, dataset_b, **parametrs_b)
        df_b = df_b.rename(columns={"Auc": "B_ind"})
        df_tl_supervised = run_supervised_transfer_cv(seed, dataset, **parametrs_tl)
        df_tl_supervised = df_tl_supervised.rename(columns={"TL_Auc": "XY_TL"})

        df_tl_unsupervised = run_unsupervised_transfer_cv(seed, dataset, **parametrs_tl_unsupervised)
        df_tl_unsupervised = df_tl_unsupervised.rename(columns={"TL_Auc": "X_TL"})

        df_tl = run_CCSA_transfer(seed, dataset_tl, **parameters_CCSA)
        df_tl = df_tl.rename(columns={"TL_Auc": "CCSA_TL"})

        df1 = pd.concat([df_m, df_w['W_ind'], df_b['B_ind'], df_tl['CCSA_TL'],
                        # df_tl_unsupervised['X_TL'],
                         df_tl_supervised['XY_TL']],
                        sort=False, axis=1)

        res = res.append(df1)

    print (res)
    res['cancer_type'] = cancer_type
    res['feature_type'] = '-'.join(feature_type)
    res['target'] = target
    res['years'] = years
    return res


def main():

    res = pd.DataFrame()
    arguments = sys.argv
    print (arguments)
    df = pd.read_csv('todo.csv', index_col='index')
    df = df.sort_index()
    todo = [63, 64, 80, 81, 84, 85, 86, 87, 90, 91, 92, 111, 116, 123, 124, 132, 156, 159, 165, 166, 167, 169, 194, 201, 205, 208, 228, 234, 254, 272, 273, 275, 303, 381, 395, 396, 400, 401, 402, 403, 404, 405, 422, 427, 431, 435, 436]
    df = df.iloc[todo]

    if not os.path.exists('./Result'):
        os.makedirs('./Result')

    for index, row in df.iterrows():
        cancer = row['cancer_type']
        feature = row['feature_type']
        target = row['target']
        years = row['years']
        try:
            print(index)
            df_m = run_cv(cancer, [feature], target, years=years, groups=("WHITE", "BLACK"))
            res = res.append(df_m)
            save_to = './Result/' + str(index) + '.xlsx'
            res.to_excel(save_to)
        except:
            continue

    # df_m = run_cv('KIRP', ['mRNA'], 'DSS', years=4, groups=("WHITE", "BLACK"))
    # res = res.append(df_m)
    print (res)


if __name__ == '__main__':
    main()
