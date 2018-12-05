import logging
import os
import os.path
import sys

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K

# x86-64 cache block size is 512-bytes and word size is 8-bytes
blocksize = np.uint64(512)
wordsize = np.uint64(8)

# Fixed maximum cluster width in cache blocks
max_cluster_width = 8
max_cluster_width_bytes = max_cluster_width * blocksize

# Maximum number of one-hot bits to encode deltas
n_delta_bits = max_cluster_width + 1

unroll = 2

def import_clusters(fname="cluster.out"):
    cluster_df = pd.read_csv(fname)
    return cluster_df

def import_tests(fname="test_data.csv"):
    test_df = pd.read_csv(fname)
    return test_df

def parse_tests(test_df, cluster_df):
    logger = logging.getLogger()

    n_cluster_bits = len(cluster_df)

    max_delta = test_df["delta"].max()
    max_delta_blocks = (max_delta + blocksize - 1) // blocksize
    logger.info("Interpreting deltas ... max_delta(%d) " \
                "max_delta_blocks(%d)", max_delta, max_delta_blocks)

    if (max_delta_blocks + 1 > n_delta_bits):
        logger.error("%d is not enough bits to encode the max block offset " \
                     "of %d", n_delta_bits, max_delta_blocks)

        sys.exit(1)

    inputs = {}
    outputs = {}
    for cluster, group in test_df.groupby("cluster"):
        clusters = np.full((len(group), 1), cluster, dtype=np.uint8)
        '''
        clusters_1h = np.zeros((len(group), n_cluster_bits), dtype=np.uint8)
        clusters_1h[:, cluster] = np.uint8(1)
        '''

        # Convert uint64 instruction addresses to unpacked bits
        iaddrs = group["iaddr"].values.reshape((-1, 1))
        '''
        iaddrs= np.unpackbits(group["iaddr"].values.view(np.uint8), axis=1)
        '''

        delta_blocks = (group["delta"].values + np.int64(blocksize - 1)) // np.int64(blocksize)

        deltas_1h = np.zeros((len(group), n_delta_bits), dtype=np.uint8)
        deltas_1h[np.arange(len(group)), delta_blocks + n_delta_bits // 2] = np.uint8(1)

        # Prefetch prediction is on the next delta so just shift the values by 1
        cluster_shifts = np.stack([clusters[i:i-unroll] for i in range(unroll)],
                                  axis=1)
        iaddr_shifts = np.stack([iaddrs[i:i-unroll] for i in range(unroll)],
                                axis=1)
        deltas_1h_shifts = np.stack([deltas_1h[i:i-unroll] for i in
                                     range(unroll)], axis=1)

        inputs[cluster] = np.concatenate([cluster_shifts, iaddr_shifts,
                                   deltas_1h_shifts], axis=2)
        outputs[cluster] = deltas_1h[unroll:]

        logger.info("Successfully encoded data for cluster %d: %d items",
                    cluster, len(group))

    return (inputs, outputs)

def create_network(n_clusters, cluster_emb_len, n_dct_iaddrs, iaddr_emb_len):
    logger = logging.getLogger()

    models = []

    for ci in range(n_clusters):
        cluster_input = keras.layers.Input(shape=(unroll, 1))
        cluster_emb = keras.layers.Embedding(n_clusters,
                                             cluster_emb_len)(cluster_input)
        cluster_emb = keras.layers.Reshape((unroll,
                                            cluster_emb_len))(cluster_emb)

        iaddr_input = keras.layers.Input(shape=(unroll, 1))
        iaddr_emb = keras.layers.Embedding(n_dct_iaddrs, iaddr_emb_len)(iaddr_input)
        iaddr_emb = keras.layers.Reshape((unroll, iaddr_emb_len))(iaddr_emb)

        delta_input = keras.layers.Input(shape=(unroll, n_delta_bits))

        input_cat = keras.layers.concatenate([cluster_emb, iaddr_emb,
                                              delta_input], axis=2)
        lstm_bits = cluster_emb_len + iaddr_emb_len + n_delta_bits
        lstm_in = keras.layers.Reshape((unroll, lstm_bits))(input_cat)

        lstm1 = keras.layers.LSTM(32, input_shape=(unroll, lstm_bits),
                                  return_sequences=True)(lstm_in)
        ldrop = keras.layers.Dropout(0.2)(lstm1)
        lstm2 = keras.layers.LSTM(32)(ldrop)
        drop = keras.layers.Dropout(0.2)(lstm2)
        fc = keras.layers.Dense(n_delta_bits)(drop)
        sigmoid = keras.layers.Activation("sigmoid")(fc)

        model = keras.models.Model(inputs=[cluster_input, iaddr_input, delta_input], outputs=sigmoid)
        logger.info(model.summary())
        model.compile(loss="categorical_crossentropy", optimizer="adam",
                      metrics=["accuracy"])

        models.append(model)

    weight_ties = dict()
    for ci in range(n_clusters):
        weight_ties[ci] = []
        for cj in range(n_clusters):
            if ci == cj:
                continue
            for Wi,Wj in zip(models[ci].trainable_weights, models[cj].trainable_weights):
                weight_ties[ci].append(tf.assign(Wj, Wi))


    return models, weight_ties

def fit_network(models, in_data_stacked, out_data_stacked):
    n_clusters = len(models)

    for cluster in range(n_clusters):
        in_data = in_data_stacked[cluster]
        out_data = out_data_stacked[cluster]

        models[cluster].fit(in_data, out_data, epochs=5, validation_split=0.05,
                            shuffle=True, verbose=1)

    return models

def predict(models, in_data, cluster_in_data):
    outputs = []

    '''
    for i in range(cluster_in_data.shape[0]):
        cluster = cluster_in_data[i]
        input_data = [(in_data[0])[i,:].reshape(1,1), (in_data[1])[i,:].reshape(1,-1)]
        output = models[cluster].predict(input_data)
        outputs.append(output)
    '''

    return np.array(outputs)

def save_weights(models, weights_dirpath, weights_prefix="weights"):
    logger = logging.getLogger()

    if not os.path.exists(weights_dirpath):
        os.makedirs(weights_dirpath)
        logger.info("Created path to weights {}.".format(weights_dirpath))

    for i in range(len(models)):
        model = models[i]
        weights_fname = weights_prefix + '_' + str(i) + '.h5'
        weights_path = os.path.join(weights_dirpath, weights_fname)

        logger.info("Saving model weights to {}".format(weights_path))
        model.save_weights(weights_path)
        logging.info("\t> Successfully saved weights to {}".format(weights_path))

def load_weights(models, n_clusters, weights_dirpath, weights_prefix="weights"):
    logger = logging.getLogger()

    for i in range(n_clusters):
        model = models[i]
        weight_fname = weights_prefix + '_' + str(i) + '.h5'
        weight_path = os.path.join(weights_dirpath, weight_fname)

        model.load_weights(weight_path)
        logging.info("\t> Successfully loaded weights from " \
                     "{}".format(weight_path))

def save_arch(models, arch_path="model.json"):
    logger = logging.getLogger()

    logger.info("Saving model architecture to {}".format(arch_path))
    fout = open(arch_path, "w")
    fout.write(models[0].to_json())
    fout.close()
    logging.info("\t> Successfully saved model architecture to " \
                 "{}".format(arch_path))

def load_arch(num_models, arch_path="model.json"):
    logger = logging.getLogger()

    fin = open(arch_path, "r")
    json = fin.read()
    fin.close()

    models = []
    for i in range(num_models):
        models.append(keras.models.model_from_json(json))
        logging.info("\t> Successfully loaded model architecture from" \
                     "{}".format(arch_path))

    return models
