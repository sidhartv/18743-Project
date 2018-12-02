import logging
import os
import sys

import keras
import numpy as np
import pandas as pd
import sklearn.preprocessing

blocksize = np.uint64(512)
wordsize = np.uint64(8)

# Maximum width of clusters in cache blocks
width = 8

def import_clusters(fname = "cluster.out"):
    cluster_df = pd.read_csv(fname)
    return cluster_df

def import_tests(fname="test_data.csv"):
    test_df = pd.read_csv(fname)
    return test_df

def parse_tests(test_df):
    logger = logging.getLogger()

    # Convert clusters to categorical representation
    cluster_lab_enc = sklearn.preprocessing.LabelEncoder()
    clusters_labeled = cluster_lab_enc.fit_transform(test_df["cluster"])

    # Convert clusters to one-hot representation
    cluster_1h_enc = sklearn.preprocessing.OneHotEncoder(sparse=False)
    cluster_1h_enc.fit(clusters_labeled.reshape(-1, 1))

    # Convert deltas to categorical representation
    delta_lab_enc = sklearn.preprocessing.LabelEncoder()
    deltas_labeled = delta_lab_enc.fit_transform(test_df["delta"])

    # Convert categorical deltas to one-hot representation
    delta_1h_enc = sklearn.preprocessing.OneHotEncoder(sparse=False)
    delta_1h_enc.fit(deltas_labeled.reshape(-1, 1))

    inputs = {}
    outputs = {}

    for name, group in test_df.groupby("cluster"):
        cluster_lab = cluster_lab_enc.transform([name])
        cluster_1h = cluster_1h_enc.transform(cluster_lab.reshape(1, -1))
        clusters_1h = np.repeat(cluster_1h, len(group), axis=0)

        # Convert uint64 instruction addresses to unpacked bits
        '''
        iaddrs= np.unpackbits(group["iaddr"].values.view(np.uint8),
                              axis=1)
        '''
        iaddrs = group["iaddr"].values

        deltas_lab = delta_lab_enc.transform(group["delta"])
        deltas_1h = delta_1h_enc.transform(deltas_lab.reshape(-1, 1))

        # Prefetch prediction is on the next delta so just shift the values by 1
        inputs[name] = np.hstack((clusters_1h[:-1],
                                  iaddrs[:-1].reshape(-1, 1),
                                  deltas_1h[:-1]))
        outputs[name] = deltas_1h[1:]

        logger.info("Successfully encoded data for cluster %d: %d items", name,
                    len(group))

    n_delta_bits = len(delta_1h_enc.categories_[0])
    return (inputs, outputs, n_delta_bits)

def create_network(n_dct_clusters, n_dct_iaddrs, iaddr_emb_len, n_delta_bits):
    logger = logging.getLogger()

    cluster_input = keras.layers.Input(shape=(1, n_dct_clusters))

    iaddr_input = keras.layers.Input(shape=(1, 1))
    iaddr_emb = keras.layers.Embedding(n_dct_iaddrs, iaddr_emb_len)(iaddr_input)

    delta_input = keras.layers.Input(shape=(1, n_delta_bits))

    input_cat = keras.layers.concatenate([cluster_input, iaddr_input,
                                          delta_input])

    lstm_bits = n_dct_clusters + n_dct_iaddrs + n_delta_bits
    lstm1 = keras.layers.LSTM(n_delta_bits,
                              input_shape=(1, lstm_bits),
                              return_sequences=True)(input_cat)
    lstm2 = keras.layers.LSTM(n_delta_bits)(lstm1)
    softmax = keras.layers.Activation("softmax")(lstm2)

    model = keras.models.Model(inputs=[cluster_input, iaddr_input, delta_input],
                               outputs=softmax)

    logger.info(model.summary())
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def fit_network(model, in_data, out_data):
    model.fit(in_data, out_data, epochs=5)

    return model

def save_weights(model, weights_fname):
    logger = logging.getLogger()
    logger.info("Saving model weights to {}".format(model_fname))
    model.save_weights(model_fname)
    logging.info("\t> Successfully saved model")

def load_weights(model, weights_fname):
    logger = logging.getLogger()
    logger.info("Loading weights from {}".format(model_fname))
    model.load_weights(weights_fname)
