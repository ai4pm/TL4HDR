from __future__ import division
from math import log, exp

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', 1000)
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

class SimulatedData:
    def __init__(self,
                 num_var=200,
                 pos=(0, 0, 0, 0),
                 neg=(0, 0, 0, 0),
                 neut=(100,0,0,100)
                 ):
        self.num_var = num_var
        self.pos = pos
        self.neg = neg
        self.neut = neut

    def generate_black_white_data(self, file_name, white_alive=400, white_dead=20, black_alive=40, black_dead=50):
        # fix the random seed for reproduction
        np.random.seed(321)
        self.white_dead = white_dead
        self.black_dead = black_dead

        path = file_name
        A = loadmat(path)
        [df_res, R] = self.get_smpled(A, num_samples_w=white_alive + white_dead, num_samples_b=black_alive + black_dead)

        Y = self.get_linear_logistc(df_res, R)
        Y = Y.astype('int32')
        X = df_res.values.astype(np.float32)
        y_sub = R
        y_strat = [str(r) + str(y) for r, y in zip(R, Y)]
        dataset = [X, Y, R, y_sub, np.asarray(y_strat)]
        return dataset

    def get_linear_logistc(self, x, R):
        beta_w, beta_b = self.get_b_by_ratio()
        x_w = x[R == 'WHITE']
        x_b = x[R == 'BLACK']

        # use a logistic regression model to get the label of each patient
        y = np.zeros((x.shape[0],))
        y_white = np.dot(x_w, beta_w)
        y_black = np.dot(x_b, beta_b)

        end_p_white = np.sort(y_white.flatten())[self.white_dead-1]
        end_p_black = np.sort(y_black.flatten())[self.black_dead-1]

        for idx in range(len(y_white)):
            y_white[idx] = 1 if y_white[idx] > end_p_white else 0

        for idx in range(len(y_black)):
            y_black[idx] = 1 if y_black[idx] > end_p_black else 0

        y[R == 'WHITE'] = y_white
        y[R == 'BLACK'] = y_black
        return y

    def get_b_by_ratio(self):

        b = np.ones(shape=(self.num_var,))
        b_w = []
        b_b = []

        b_w_pos, b_w_neg, b_w_neut = [], [], []
        b_b_pos, b_b_neg, b_b_neut = [], [], []

        w_pos = self.pos
        w_neg = self.neg
        w_neut = self.neut

        mask = [[-1, -1], [-1, 1], [1, -1],  [1, 1]]
        for idx, val in enumerate(mask):
            b_w_pos.extend([val[0]] * w_pos[idx])
            b_w_neg.extend([val[0]] * w_neg[idx])
            b_w_neut.extend([val[0]] * w_neut[idx])

            b_b_pos.extend([val[1]] * w_pos[idx])
            b_b_neg.extend([val[1]] * w_neg[idx])
            b_b_neut.extend([val[1]] * w_neut[idx])

        b_w.extend(b_w_pos)
        b_w.extend(b_w_neg)
        b_w.extend(b_w_neut)

        b_b.extend(b_b_pos)
        b_b.extend(b_b_neg)
        b_b.extend(b_b_neut)

        b_w = b_w * b
        b_b = b_b * b
        return [b_w, b_b]

    def get_smpled(self, A, num_samples_w=400, num_samples_b=40):
        # data generated from the R package, so we need more subscripts to unpack it
        data = A['data']
        de = data['de'][0][0]
        de = np.squeeze(de)
        group = data['group'][0][0]
        group = np.squeeze(group) - 1
        counts = data['counts'][0][0]
        counts = counts.transpose()
        nGenes = counts.shape[1]
        df = pd.DataFrame(counts, columns=range(nGenes))

        df_col = pd.DataFrame(de, index=df.columns, columns=['de'])
        df_col_pos_idx = list(df_col[df_col['de'] == 1].index)
        df_col_neg_idx = list(df_col[df_col['de'] == -1].index)
        df_col_neut_idx = list(df_col[df_col['de'] == 0].index)

        genes_kept = df_col_pos_idx
        genes_kept.extend(df_col_neg_idx)
        genes_kept.extend(df_col_neut_idx)

        df = df[genes_kept]
        df['group'] = pd.Series(group, index=df.index)
        df['group'] = df['group'].map(lambda x: 'BLACK' if x == 1 else 'WHITE')
        df_w = df[df['group'] == 'WHITE']
        df_b = df[df['group'] == 'BLACK']

        df_w_kept = df_w.sample(n=num_samples_w)
        df_b_kept = df_b.sample(n=num_samples_b)

        df_res = pd.concat([df_w_kept, df_b_kept])
        R = df_res['group'].values
        df_res.drop(columns=['group'], inplace=True)

        X = df_res.values
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = pd.DataFrame(X)
        return [X, R]