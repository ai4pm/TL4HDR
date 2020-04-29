import theano
from keras import Input, Model
from keras.layers import Dropout, Dense, Activation, Lambda
from keras.optimizers import SGD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut

from data.preProcess import get_n_years
import pandas as pd
import numpy as np

from model.CCSA import Initialization
from model.mlp import get_k_best, MLP


def run_cv(seed, fold, X, Y, R, y_strat, val_size=0, pretrain_set=None, batch_size=32, k=-1,
           learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, momentum=0.9,
           L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128,64]):

    X_w = pretrain_set.get_value(borrow=True) if k > 0 and pretrain_set else None

    m = X.shape[1] if k < 0 else k
    columns = list(range(m))
    columns.extend(['scr', 'R', 'Y'])
    df = pd.DataFrame(columns=columns)
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X, y_strat):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        R_train, R_test = R[train_index], R[test_index]
        strat_train, strat_test = y_strat[train_index], y_strat[test_index]

        if k > 0:
            k_best = SelectKBest(f_classif, k=k)
            k_best.fit(X_train, Y_train)
            X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)

            if pretrain_set:
                X_base = k_best.transform(X_w)
                pretrain_set = theano.shared(X_base, name='pretrain_set', borrow=True)

        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                              test_size=val_size, random_state=0,
                                                              stratify=strat_train)
            valid_data = (X_val, Y_val)
        train_data = (X_train, Y_train)

        n_in = X_train.shape[1]
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers, momentum=momentum)
        if pretrain_set:
            pretrain_config = {'pt_batchsize': 32, 'pt_lr': 0.01, 'pt_epochs': 500, 'corruption_level': 0.3}
            classifier.pretrain(pretrain_set=pretrain_set, pretrain_config=pretrain_config)
            classifier.tune(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        else:
            classifier.train(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        X_scr = classifier.get_score(X_test)

        array1 = np.column_stack((X_test, X_scr[:,1], R_test, Y_test))
        df_temp1 = pd.DataFrame(array1, index=list(test_index), columns=columns)
        df = df.append(df_temp1)

    return df

def run_mixture_cv(seed, dataset, fold=3, k=-1, val_size=0, batch_size=32, momentum=0.9,
                   learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, save_to=None,
                   L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64], groups=("WHITE", "BLACK")):
    X, Y, R, y_sub, y_strat = dataset
    df = run_cv(seed, fold, X, Y, R, y_strat, val_size=val_size, batch_size=batch_size, k=k, momentum=momentum,
                learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout, n_epochs=n_epochs,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    if save_to:
        df.to_csv(save_to)
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    y_test_w, y_scr_w = list(df.loc[df['R']==groups[0], 'Y'].values), \
                        list(df.loc[df['R']==groups[0], 'scr'].values)
    y_test_b, y_scr_b = list(df.loc[df['R']==groups[1], 'Y'].values), \
                        list(df.loc[df['R']==groups[1], 'scr'].values)

    A_CI, W_CI, B_CI = roc_auc_score(y_test, y_scr, average='weighted'), \
                       roc_auc_score(y_test_w, y_scr_w, average='weighted'), \
                       roc_auc_score(y_test_b, y_scr_b, average='weighted')

    res = {'folds': fold, 'A_Auc': A_CI,
           'W_Auc': W_CI, 'B_Auc': B_CI}
    df = pd.DataFrame(res, index=[seed])
    return df

def run_one_race_cv(seed, dataset, fold=3,  k=-1, val_size=0, batch_size=32,
                    learning_rate=0.01, lr_decay=0.0, dropout=0.5, save_to=None,
                    L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):
    X, Y, R, y_sub, y_strat = dataset
    df = run_cv(seed, fold, X, Y, R, y_strat, val_size=val_size, batch_size=batch_size, k=k,
                learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    if save_to:
        df.to_csv(save_to)
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'folds': fold, 'Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
    return df

def run_CCSA_transfer(seed, dataset, n_features, fold=3, alpha=0.25, learning_rate = 0.01,
                      hiddenLayers=[100, 50], dr=0.5, groups=("WHITE", "BLACK"),
                      momentum=0.0, decay=0, batch_size=32,
                      sample_per_class=2, repetition=1):
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(X)
    df['R'] = R
    df['Y'] = Y

    df_train = df[df['R'] == groups[0]]
    df_w_y = df_train['Y']
    df_train = df_train.drop(columns=['Y', 'R'])

    Y_train_source = df_w_y.values.ravel()
    X_train_source = df_train.values

    df_test = df[df['R'] == groups[1]]
    df_b_y = df_test['Y']
    df_test = df_test.drop(columns=['Y', 'R'])

    Y_test = df_b_y.values.ravel()
    X_test = df_test.values

    if n_features > 0 and n_features < X_test.shape[1]:
        X_train_source, X_test = get_k_best(X_train_source, Y_train_source, X_test, n_features)
    else:
        n_features = X_test.shape[1]

    df_score = pd.DataFrame(columns=['scr', 'Y'])
    kf = StratifiedKFold(n_splits=fold, random_state=seed)
    for train_index, test_index in kf.split(X_test, Y_test):
        X_train_target_full, X_test_target = X_test[train_index], X_test[test_index]
        Y_train_target_full, Y_test_target = Y_test[train_index], Y_test[test_index]

        index0 = np.where(Y_train_target_full == 0)
        index1 = np.where(Y_train_target_full == 1)

        target_samples = []
        target_samples.extend(index0[0][0:sample_per_class])
        target_samples.extend(index1[0][0:sample_per_class])

        X_train_target = X_train_target_full[target_samples]
        Y_train_target = Y_train_target_full[target_samples]

        X_val_target = [e for idx, e in enumerate(X_train_target_full) if idx not in target_samples]
        Y_val_target = [e for idx, e in enumerate(Y_train_target_full) if idx not in target_samples]

        best_score, best_Auc = train_and_predict(X_train_target, Y_train_target,
                                     X_train_source, Y_train_source,
                                     X_val_target, Y_val_target,
                                     X_test_target, Y_test_target,
                                     sample_per_class=sample_per_class,
                                     alpha=alpha, learning_rate=learning_rate,
                                     hiddenLayers=hiddenLayers, dr=dr,
                                     momentum=momentum, decay=decay,
                                     batch_size=batch_size,
                                     repetition=repetition,
                                     n_features=n_features)

        print (best_score.shape)
        print (Y_test_target.shape)

        array = np.column_stack((best_score, Y_test_target))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'Y'])
        df_score = df_score.append(df_temp)

    auc = roc_auc_score(df_score['Y'].values, df_score['scr'].values)
    res = {'TL_Auc': auc}
    df = pd.DataFrame(res, index=[seed])
    return df


# let's run the experiments when 1 target sample per calss is available in training.
# you can run the experiments for sample_per_class=1, ... , 7.
# sample_per_class = 2
# Running the experiments for repetition 5. In the paper we reported the average acuracy.
# We run the experiments for repetition=0,...,9 and take the average
# repetition = 2
def train_and_predict(X_train_target, y_train_target,
                      X_train_source, y_train_source,
                      X_val_target, Y_val_target,
                      X_test, y_test,
                      repetition, sample_per_class,
                      alpha=0.25, learning_rate = 0.01,
                      hiddenLayers=[100, 50], dr=0.5,
                      momentum=0.0, decay=0, batch_size=32,
                      n_features = 400):
    # size of input variable for each patient
    domain_adaptation_task = 'WHITE_to_BLACK'
    input_shape = (n_features,)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # number of classes for digits classification
    nb_classes = 2
    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = alpha

    # Having two streams. One for source and one for target.
    model1 = Initialization.Create_Model(hiddenLayers=hiddenLayers, dr=dr)
    processed_a = model1(input_a)
    processed_b = model1(input_b)

    # Creating the prediction function. This corresponds to h in the paper.
    processed_a = Dropout(0.5)(processed_a)
    out1 = Dense(nb_classes)(processed_a)
    out1 = Activation('softmax', name='classification')(out1)

    distance = Lambda(Initialization.euclidean_distance, output_shape=Initialization.eucl_dist_output_shape,
                      name='CSA')([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay) # momentum=0., decay=0.
    model.compile(loss={'classification': 'binary_crossentropy', 'CSA': Initialization.contrastive_loss},
                  optimizer=optimizer,
                  loss_weights={'classification': 1 - alpha, 'CSA': alpha})

    print('Domain Adaptation Task: ' + domain_adaptation_task)
    # for repetition in range(10):
    Initialization.Create_Pairs(domain_adaptation_task, repetition, sample_per_class,
                                X_train_target, y_train_target, X_train_source, y_train_source,
                                n_features=n_features)
    best_score, best_Auc = Initialization.training_the_model(model, domain_adaptation_task, repetition, sample_per_class,batch_size,
                                                             X_val_target, Y_val_target,
                                                             X_test, y_test)

    print('Best AUC for {} target sample per class and repetition {} is {}.'.format(sample_per_class,
                                                                                             repetition, best_Auc))
    return best_score, best_Auc


def run_unsupervised_transfer_cv(seed, dataset, fold=3, val_size=0, k=-1, batch_size=32, save_to=None,
                    learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, target='BLACK',
                    L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):
    X, Y, R, y_sub, y_strat = dataset
    idx = R == target
    X_b, y_b, R_b, y_strat_b = X[idx], Y[idx], R[idx], y_strat[idx]
    pretrain_set = theano.shared(X, name='pretrain_set', borrow=True)

    df = run_cv(seed, fold, X_b, y_b, R_b, y_strat_b, pretrain_set=pretrain_set,
                val_size=val_size, batch_size=batch_size, k=k, n_epochs=n_epochs,
                learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    if save_to:
        df.to_csv(save_to)
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'folds': fold, 'TL_Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
    return df

def run_supervised_transfer_cv(seed, dataset, fold=3, val_size=0, k=-1, batch_size=32, groups=('WHITE', 'BLACK'),
                    learning_rate=0.01, lr_decay=0.0, dropout=0.5, tune_epoch=200, tune_lr=0.002, train_epoch=1000,
                    L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64], tune_batch=10):
    X, Y, R, y_sub, y_strat = dataset
    idx = R == groups[1]
    X_b, y_b, R_b, y_strat_b = X[idx], Y[idx], R[idx], y_strat[idx]
    idx = R == groups[0]
    X_w, y_w, R_w, y_strat_w = X[idx], Y[idx], R[idx], y_strat[idx]
    pretrain_set = (X_w, y_w)

    df = pd.DataFrame(columns=['scr', 'R', 'Y'])
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X_b, y_strat_b):
        X_train, X_test = X_b[train_index], X_b[test_index]
        Y_train, Y_test = y_b[train_index], y_b[test_index]
        R_train, R_test = R_b[train_index], R_b[test_index]
        strat_train, strat_test = y_strat_b[train_index], y_strat_b[test_index]

        if k > 0:
            k_best = SelectKBest(f_classif, k=k)
            k_best.fit(X_train, Y_train)
            X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)
            X_base = k_best.transform(X_w)
            pretrain_set = (X_base, y_w)

        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                              test_size=val_size, random_state=0,
                                                              stratify=strat_train)
            valid_data = (X_val, Y_val)
        train_data = (X_train, Y_train)

        n_in = X_train.shape[1]
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers)
        classifier.train(pretrain_set, n_epochs=train_epoch, batch_size=batch_size)
        classifier.learning_rate = tune_lr
        classifier.tune(train_data, valid_data=valid_data, batch_size=tune_batch, n_epochs=tune_epoch)

        scr = classifier.get_score(X_test)
        array = np.column_stack((scr[:, 1], R_test, Y_test))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'R', 'Y'])
        df = df.append(df_temp)

    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'folds': fold, 'TL_Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
    return df
