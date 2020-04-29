from __future__ import division
import timeit
import pandas as pd
import numpy as np
import scipy
from scipy.io import loadmat, savemat
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from mlxtend.evaluate import permutation_test
from scipy.stats import stats

pd.set_option('display.width', 1000)
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

def permutation_ttest(W, B):
    p_value = permutation_test(W, B,
                               method='approximate',
                               num_rounds=100,
                               func=lambda W, B: stats.ttest_ind(W, B),
                               seed=0)
    return 1 if p_value < 0.05 else 0


def get_logistic(f_name, feature_type, groups=['WHITE', 'BLACK']):

    data = loadmat(f_name)
    X, Y, R = data['X'], data['Y'][0], data['R'][0]

    Y_W = Y[R == groups[0]]
    Y_B = Y[R == groups[1]]
    AA_A, AA_D = sum(Y_B == 1),  sum(Y_B == 0)
    EA_A, EA_D = sum(Y_W == 1), sum(Y_W == 0)
    res = {}
    EA = sum(R == groups[0])
    AA = sum(R == groups[1])
    res['AA-ratio'] = AA / (AA + EA )
    res['AA-Positive'] = AA_A
    res['AA-Negative'] = AA_D
    res['EA-Positive'] = EA_A
    res['EA-Negative'] = EA_D

    if feature_type != 'Protein':
        k_best = SelectKBest(f_classif, k=200)
        k_best.fit(X, Y)
        X = k_best.transform(X)

    R = [str(row[0]) for row in R]
    R = np.asarray(R)
    X_W, Y_W = X[R == groups[0]], Y[R == groups[0]]
    X_B, Y_B = X[R == groups[1]], Y[R == groups[1]]

    n = X.shape[1]
    ttest = 0
    for j in range(n):
        x_w, x_b = X_W[:,j], X_B[:,j]
        ttest = ttest + permutation_ttest(x_w, x_b)
    res['ttest'] = ttest / n

    logreg = LogisticRegression()
    logreg.fit(X_W, Y_W)
    coef_w = logreg.coef_[0]
    logreg1 = LogisticRegression()
    logreg1.fit(X_B, Y_B)
    coef_b = logreg1.coef_[0]

    corr = scipy.stats.pearsonr(coef_w, coef_b)
    res['corr'] = corr[0]
    key = f_name
    df1 = pd.DataFrame(res, index=[key])
    return df1

def main():

    start_time = timeit.default_timer()
    print (start_time)
    df = pd.read_excel('223Examples.xlsx', index_col='index')
    df = df.sort_index()

    res = pd.DataFrame()
    for index, row in df.iterrows():
        cancer = row['cancer_type']
        feature = row['feature_type']
        target = row['target']
        years = row['years']
        f_name = cancer + '-AA-EA-' + feature + '-' + target + '-' + years + '-YR'
        df = get_logistic(f_name, [feature], target)
        df['GAP'] = row['GAP']
        res = res.append(df)

    end_time = timeit.default_timer()
    print('Pretraining took {} minutes.'.format((end_time - start_time) / 60.))
    print (res)

if __name__ == '__main__':
    main()
