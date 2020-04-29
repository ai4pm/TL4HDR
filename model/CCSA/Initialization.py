import random
import os

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Nadam
from keras import backend as K
import numpy as np
import sys
import pandas as pd
from scipy.io import loadmat
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler



def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def get_k_best(X_train, Y_train, k):
    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(X_train, Y_train)

    return k_best

def Create_Pairs(domain_adaptation_task,repetition,sample_per_class,
                 X_train_target, y_train_target, X_train_source, y_train_source,
                 n_features=400):

    UM  = domain_adaptation_task
    cc  = repetition
    SpC = sample_per_class

    print ('Creating pairs for repetition: '+str(cc)+' and sample_per_class: '+str(sample_per_class))
    Training_P=[]
    Training_N=[]

    for trs in range(len(y_train_source)):
        for trt in range(len(y_train_target)):
            if y_train_source[trs]==y_train_target[trt]:
                Training_P.append([trs,trt])
            else:
                Training_N.append([trs,trt])


    random.shuffle(Training_N)
    Training = Training_P+Training_N[:3*len(Training_P)]
    random.shuffle(Training)

    X1=np.zeros([len(Training),n_features],dtype='float32')
    X2=np.zeros([len(Training),n_features],dtype='float32')

    y1=np.zeros([len(Training)])
    y2=np.zeros([len(Training)])
    yc=np.zeros([len(Training)])

    for i in range(len(Training)):
        in1,in2=Training[i]
        X1[i,:]=X_train_source[in1,:]
        X2[i,:]=X_train_target[in2,:]

        y1[i]=y_train_source[in1]
        y2[i]=y_train_target[in2]
        if y_train_source[in1]==y_train_target[in2]:
            yc[i]=1

    if not os.path.exists('./CCSA_pairs'):
        os.makedirs('./CCSA_pairs')

    np.save('./CCSA_pairs/' + UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', X1)
    np.save('./CCSA_pairs/' + UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', X2)

    np.save('./CCSA_pairs/' + UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', y1)
    np.save('./CCSA_pairs/' + UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', y2)
    np.save('./CCSA_pairs/' + UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', yc)


def Create_Model(hiddenLayers=[100, 50], dr=0.5):

    model = Sequential()
    for idx, nodes in enumerate(hiddenLayers):
        model.add(Dense(nodes))
        model.add(Activation('relu'))
        if dr > 0:
            model.add((Dropout(0.5)))
    return model


def euclidean_distance(vects):
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def training_the_model(model,domain_adaptation_task,repetition,sample_per_class, batch_size,
                        X_val_target, Y_val_target,
                        X_test, y_test):
    nb_classes=2
    UM = domain_adaptation_task
    cc = repetition
    SpC = sample_per_class

    epoch = 100  # Epoch number
    batch_size = batch_size
    y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_val_target = np_utils.to_categorical(Y_val_target, nb_classes)


    X1 = np.load('./CCSA_pairs/' + UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    X2 = np.load('./CCSA_pairs/' + UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')

    y1 = np.load('./CCSA_pairs/' + UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    y2 = np.load('./CCSA_pairs/' + UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    yc = np.load('./CCSA_pairs/' + UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')

    y1 = np_utils.to_categorical(y1, nb_classes)
    y2 = np_utils.to_categorical(y2, nb_classes)

    print ('Training the model - Epoch '+str(epoch), ' total trainings:', X1.shape, X2.shape)
    nn=batch_size
    best_Acc = 0
    best_Auc = 0
    best_score = []

    for e in range(epoch):
        if e % 10 == 0:
            printn(str(e) + '->')
        for i in range(len(y2) // nn):
            loss = model.train_on_batch([X1[i * nn:(i + 1) * nn, :], X2[i * nn:(i + 1) * nn, :]],
                                        [y1[i * nn:(i + 1) * nn, :], yc[i * nn:(i + 1) * nn, ]])
            loss1 = model.train_on_batch([X2[i * nn:(i + 1) * nn, :], X1[i * nn:(i + 1) * nn, :]],
                                        [y2[i * nn:(i + 1) * nn, :], yc[i * nn:(i + 1) * nn, ]])

        # Out = model.predict([X_test, X_test])
        # Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_test, axis=1)
        # Acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)
        # Auc = roc_auc_score(y_test[:,1], Out[0][:,1])

        # if best_Acc < Acc:
        #     best_Acc = Acc
        #     best_score = Out[0][:,1]
        # if best_Auc < Auc:
        #     best_Auc = Auc
        #     best_score = Out[0][:, 1]
        # fine tune the model to get a better performance
    for e in range(epoch):
        model.train_on_batch([X_val_target, X_val_target], [Y_val_target, Y_val_target])
    Out = model.predict([X_test, X_test])
    score = Out[0][:,1]
    Auc = roc_auc_score(y_test[:, 1], Out[0][:, 1])

    print (str(e))
    return score, Auc