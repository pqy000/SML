import os
import numpy as np 
import pickle
import sys 
import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.sparse as sp
sys.path.append('AFGSM')
import GCN
import utils

adj_path = './Data/experimental_adj.pkl'
feature_path = './Data/experimental_features.pkl'
label_path = './Data/experimental_train.pkl'

def read_data(adj_path, feature_path, label_path):
    with open(adj_path, 'rb') as f:
        _A_obs = pickle.load(f)
    with open(feature_path, 'rb') as f:
        _X_obs = pickle.load(f)
    with open(label_path, 'rb') as f:
        _z_obs = pickle.load(f)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"
    _An = utils.preprocess_graph(_A_obs)
    _K = _z_obs.max() + 1
    _Z_obs = np.eye(_K)[_z_obs]
    sizes = [64, _K]
    return _An, _X_obs, _Z_obs, sizes

def train_gcn(_An, _X_obs, _Z_obs, sizes, gpu_id=0):
    split_train = [i for i in range(493486)]
    split_val = [i + 493486 for i in range(50000)]
    split_test = [i + 543486 for i in range(50000)]
    with tf.Graph().as_default():
        surrogate_model = GCN.GCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
        #surrogate_model.train(_An, _X_obs, split_train, split_val, _Z_obs, n_iters=400)
        surrogate_model.train(split_train, split_val, _Z_obs, n_iters=1000)
        '''
        W1 = surrogate_model.W1.eval(session=surrogate_model.session)
        W2 = surrogate_model.W2.eval(session=surrogate_model.session)
        logits = surrogate_model.logits.eval(session = surrogate_model.session)
        '''
        surrogate_model.session.close()

if __name__ == "__main__":
    _An, _X_obs, _Z_obs, sizes = read_data(adj_path, feature_path, label_path)
    train_gcn(_An, _X_obs, _Z_obs, sizes, gpu_id=0)