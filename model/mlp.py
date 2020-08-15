import time
import timeit
import lasagne
import numpy
import theano
import theano.tensor as T
from lasagne.nonlinearities import rectify
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from model.DropoutLayer import DropoutHiddenLayer
from model.HiddenLayer import HiddenLayer
from model.LogisticRegression import LogisticRegression
from model.SdA import dA


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self, n_in,
             learning_rate, hidden_layers_sizes=None,
             lr_decay=0.0, momentum=0.9,
             L2_reg=0.0, L1_reg=0.0,
             activation="rectify",
             dropout=None,
             batch_norm=False,
             standardize=False,
             numpy_rng=None,
             theano_rng=None):
        self.X = T.fmatrix('X')  # patients covariates
        self.y = T.ivector('Y')  # the observations vector
        self.is_train = T.iscalar('is_train')

        ## sdA_hidden_layers, a HiddenLayer object is used to store the layers shared by stacked auto-encoders
        self.hidden_layers = []
        ## where to store the auto-encoder
        self.dA_layers = []
        self.n_layers = len(hidden_layers_sizes)
        self.hidden_layers_sizes = hidden_layers_sizes
        self.L1 = 0
        self.L2 = 0

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(11111)
        activation_fn = rectify
        self.params = []

        for idx, hidden_layer_size in enumerate(hidden_layers_sizes):
            if idx == 0:
                input_size = n_in
                layer_input = self.X
            else:
                input_size = hidden_layers_sizes[idx - 1]
                layer_input = self.hidden_layers[-1].output

            if dropout and dropout > 0:
                hidden_layer = DropoutHiddenLayer(rng=numpy_rng,
                                                  input=layer_input,
                                                  n_in=input_size,
                                                  n_out=hidden_layers_sizes[idx],
                                                  activation=activation_fn,
                                                  dropout_rate=dropout,
                                                  is_train=self.is_train)
            else:
                hidden_layer = HiddenLayer(rng=numpy_rng,
                                           input=layer_input,
                                           n_in=input_size,
                                           n_out=hidden_layers_sizes[idx],
                                           activation=activation_fn)

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[idx],
                          W=hidden_layer.W,
                          bhid=hidden_layer.b,
                          non_lin=activation_fn)
            self.dA_layers.append(dA_layer)

            self.L1 += abs(hidden_layer.W).sum()
            self.L2 += (hidden_layer.W ** 2).sum()

        # Adds a risk prediction layer on top of the stack.
        if self.n_layers == 0:
            self.logRegressionLayer = LogisticRegression(
                input=self.X,
                n_in=n_in,
                n_out=2
            )
        else:
            self.logRegressionLayer = LogisticRegression(
                input=self.hidden_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=2)
        self.L1 += abs(self.logRegressionLayer.W).sum()
        self.L2 += (self.logRegressionLayer.W ** 2).sum()
        self.params.extend(self.logRegressionLayer.params)

        self.regularizers = L1_reg * self.L1 + L1_reg * self.L2
        self.n_in = n_in
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.momentum = momentum
        self.dropout = dropout

        self.finetune_cost = self.logRegressionLayer.negative_log_likelihood(self.y)
        self.errors = self.logRegressionLayer.errors
        self.hyperparams = {
            'n_in': n_in,
            'learning_rate': learning_rate,
            'hidden_layers_sizes': hidden_layers_sizes,
            'lr_decay': lr_decay,
            'momentum': momentum,
            'L2_reg': L2_reg,
            'L1_reg': L1_reg,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'standardize': standardize
        }

    def pretraining_functions(self, pretrain_x, batch_size):
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption
        learning_rate = T.scalar('lr')  # learning rate

        pretrain_x_batch = pretrain_x
        if batch_size:
            # begining of a batch, given `index`
            batch_begin = index * batch_size
            # ending of a batch given `index`
            batch_end = batch_begin + batch_size
            pretrain_x_batch = pretrain_x[batch_begin: batch_end]

        pretrain_fns = []
        is_train = numpy.cast['int32'](0)   # value does not matter
        for dA_layer in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA_layer.get_cost_updates(corruption_level,
                                                      learning_rate)
            # compile the theano function
            fn = theano.function(
                on_unused_input='ignore',
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.X: pretrain_x_batch,
                    self.is_train: is_train
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def pretrain(self, pretrain_set, pretrain_config=None, verbose=False):
        n_layers = len(self.dA_layers)
        if pretrain_config is not None:
            n_batches = pretrain_set.get_value(borrow=True).shape[0]
            n_batches //= pretrain_config['pt_batchsize']

            pretraining_fns = self.pretraining_functions(
                pretrain_set,
                pretrain_config['pt_batchsize'])
            start_time = timeit.default_timer()
            # de-noising level
            corruption_levels = [pretrain_config['corruption_level']] * n_layers
            for i in range(n_layers):  # Layerwise pre-training
                # go through pretraining epochs
                for epoch in range(pretrain_config['pt_epochs']):
                    # go through the training set
                    c = []
                    for batch_index in range(n_batches):
                        c.append(pretraining_fns[i](index=batch_index,
                                                    corruption=corruption_levels[i],
                                                    lr=pretrain_config['pt_lr']))

                    if verbose:
                        print ('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c, dtype='float64')))

            end_time = timeit.default_timer()
            if verbose:
                print('Pretraining took {} minutes.'.format((end_time - start_time) / 60.))

    def build_finetune_functions(self, learning_rate, update_fn, istune=False):

        is_train = T.iscalar('is_train')
        X = T.matrix('X', dtype='float32')
        y = T.ivector('Y')

        loss = self.finetune_cost + self.regularizers
        if istune:
            loss = self.finetune_cost
        updates = update_fn(loss, self.params, learning_rate=learning_rate)
        if self.momentum:
            updates = lasagne.updates.apply_nesterov_momentum(updates,
                self.params, momentum=self.momentum)

        test = theano.function(
            on_unused_input='ignore',
            inputs=[X, y, is_train],
            outputs=[self.finetune_cost, self.errors(self.y),
                     self.logRegressionLayer.output, self.logRegressionLayer.input],
            givens={
                self.X: X,
                self.y: y,
                self.is_train: is_train
            },
            name='test'
        )

        train = theano.function(
            on_unused_input='ignore',
            inputs=[X, y, is_train],
            outputs=[self.finetune_cost, self.errors(y),
                     self.logRegressionLayer.output, self.logRegressionLayer.input],
            updates=updates,
            givens={
                self.X: X,
                self.y: y,
                self.is_train: is_train
            },
            name='train'
        )


        return train, test

    def get_params(self):
        return [param.copy().eval() for param in self.params]

    def reset_weight(self, params):
        for i in range(self.n_layers):
            self.hidden_layers[i].reset_weight((params[2 * i], params[2 * i + 1]))
        self.logRegressionLayer.reset_weight(params[-2:])

    def train(self,
              train_data, valid_data=None,
              n_epochs=500,
              validation_frequency=250,
              patience=2000, improvement_threshold=0.99999, patience_increase=2,
              batch_size=20,
              update_fn=lasagne.updates.sgd,
              **kwargs):

        x_train, y_train = train_data
        n_train_batches = x_train.shape[0]
        n_train_batches //= batch_size

        n_val_batches = n_train_batches
        if valid_data:
            x_valid, y_valid = valid_data
            n_val_batches = x_valid.shape[0]
            n_val_batches //= batch_size

        best_validation_loss = numpy.inf
        best_params = None

        # Initialize Training Parameters
        lr = theano.shared(numpy.asarray(self.learning_rate,
                                    dtype = numpy.float64))
        # momentum = numpy.array(0, dtype= numpy.float32)

        train_fn, valid_fn = self.build_finetune_functions(
            learning_rate=lr,
            update_fn=update_fn
        )

        def train_batch(index):
            x_train_batch = x_train[index * batch_size:(index + 1) * batch_size]
            y_train_batch = y_train[index * batch_size:(index + 1) * batch_size]
            cost, err, output, input = train_fn(x_train_batch, y_train_batch, 1)
            return err

        def validate_model():
            res = []
            for index in range(n_val_batches):
                x_train_batch = x_valid[index * batch_size:(index + 1) * batch_size]
                y_train_batch = y_valid[index * batch_size:(index + 1) * batch_size]
                cost, errs, output, input = train_fn(x_train_batch, y_train_batch, 0)
                res.append(errs)
            return res

        start = time.time()
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_batch(minibatch_index)

                iter = (epoch - 1) * n_train_batches + minibatch_index
                if valid_data and (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_params = [param.copy().eval() for param in self.params]
                        best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break

            decay_learning_rate = theano.function(
                inputs=[], outputs=lr,
                updates={lr: lr * (1 / (1 + self.lr_decay))})
            decay_learning_rate()

        if valid_data and best_params:
            for idx, param in enumerate(self.params):
                param.set_value(best_params[idx])

    def tune(self,
              train_data, valid_data=None,
              n_epochs=500,
              validation_frequency=250,
              batch_size=32,
              update_fn=lasagne.updates.sgd,
              **kwargs):

        x_train, y_train = train_data

        batch_size_new = min(x_train.shape[0], batch_size)
        n_train_batches = x_train.shape[0]
        n_train_batches //= batch_size_new
        best_params = None

        # Initialize Training Parameters
        lr = theano.shared(numpy.asarray(self.learning_rate,
                                    dtype = numpy.float64))
        # momentum = numpy.array(0, dtype= numpy.float32)

        train_fn, valid_fn = self.build_finetune_functions(
            learning_rate=lr,
            update_fn=update_fn,
            istune=True
        )

        def train_batch(index):
            x_train_batch = x_train[index * batch_size:(index + 1) * batch_size]
            y_train_batch = y_train[index * batch_size:(index + 1) * batch_size]
            cost, err, output, input = train_fn(x_train_batch, y_train_batch, 1)
            return err

        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_batch(minibatch_index)

            decay_learning_rate = theano.function(
                inputs=[], outputs=lr,
                updates={lr: lr * (1 / (1 + self.lr_decay))})
            decay_learning_rate()


    def get_score(self, X, is_train=0):

        score = theano.function(
            on_unused_input='ignore',
            inputs=[self.X, self.is_train],
            outputs=self.logRegressionLayer.output,
            name='score'
        )

        return score(X, is_train)

    def get_auc(self, test_data, race=None):
        x_test, y_test, r_test = test_data
        y_scr = self.get_score(x_test)[:,1]
        y_ture = y_test
        if race:
            idx = r_test == race
            y_scr = y_scr[idx]
            y_ture = y_test[idx]

        return roc_auc_score(list(y_ture), list(y_scr))


def get_k_best(X_train, y_train, X_test, k=400):
    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(X_train, y_train)
    res = (k_best.transform(X_train),
           k_best.transform(X_test))
    return res
