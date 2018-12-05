import logging
import os.path

import keras
import pandas as pd
import argparse
import sklearn.preprocessing
import numpy as np

import librnn as lr

# x86-64 cache block size is 512-bytes and word size is 8-bytes
blocksize = np.uint64(512)
wordsize = np.uint64(8)

# Fixed maximum cluster width in cache blocks
max_cluster_width = 8
max_cluster_width_bytes = max_cluster_width * blocksize

# Maximum number of one-hot bits to encode deltas
n_delta_bits = max_cluster_width + 1

def parse_args():
    parser = argparse.ArgumentParser(description="Run RNN")
    parser.add_argument("--test_data", type=str, default="test_data.csv",
                        help="Input Pandas DataFrame to parse for data (CSV format)")
    parser.add_argument("--cluster_data", type=str, default="cluster.csv",
                        help="Input Pandas DataFrame to parse for clusters (CSV format)")

    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--weights_dir", type=str, default="weights/",
                        help="Output weights file (HDF5 format)")
    parser.add_argument("--arch_file", type=str, default="model.json",
                        help="Output architecture file (JSON format)")

    return parser.parse_args()

def create_and_train(args):
    logger = logging.getLogger()

    test_df = lr.import_tests(args.test_data)
    cluster_df = lr.import_clusters(args.cluster_data)
    logger.info("Finished loading test data and cluster information.")
    in_data, out_data = lr.parse_tests(test_df, cluster_df)

    n_clusters = len(cluster_df)
    n_dct_iaddrs = test_df["iaddr"].nunique()

    iaddr_emb_len = 4
    models,weight_tie = lr.create_network(n_clusters, n_dct_iaddrs,
                                          iaddr_emb_len)

    '''
    times, features = in_data_stacked.shape
    cluster_inputs = in_data_stacked[:, 0:n_clusters].reshape(times, n_clusters)

    cluster_inputs = np.argmax(cluster_inputs, axis=1)
    iaddr_inputs = in_data_stacked[:, n_clusters].reshape(times, 1, 1)
    delta_inputs = in_data_stacked[:, n_clusters+1:].reshape(times, 1, n_delta_bits)

    trained_models = lr.fit_network(models,
            [iaddr_inputs, delta_inputs],
            cluster_inputs,
            out_data_stacked,
            weight_tie)
    '''
    trained_models = models

    if not args.no_save and args.weights_dir:
        lr.save_weights(trained_models, args.weights_dir,
                        weights_prefix="weights")

    if not args.no_save and args.arch_file:
        lr.save_arch(trained_models, args.arch_file)

def load_model(num_clusters, arch_fname="model.json", weights_prefix="weights"):
    model = lr.load_arch(num_clusters, arch_fname)
    lr.load_weights(model, num_clusters, weights_fname)

    return model

def main():
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)

    args = parse_args()
    create_and_train(args)

def init(arch_epath, weights_edirpath, cluster_epath):
    arch_path = os.path.expandvars(arch_epath)
    weights_path = os.path.expandvars(weights_epath)
    cluster_path = os.path.expandvars(cluster_epath)

    cluster_df = lr.import_clusters(cluter_path)
    n_models = len(cluster_df)

    models = lr.load_arch(n_models, arch_path)
    lr.load_weights(models, n_models, weights_edirpath,
                    weights_prefix="weights")

    return (cluster_df, models)

def chkinfer(iaddr, daddr_ual):
    daddr = np.uint64(daddr_ual) & ~np.uint64(blocksize - 1)

def infer(iaddr, daddr_ual):
    daddr = np.uint64(daddr_ual) & ~np.uint64(blocksize - 1)

    found_cluster = -1
    found_delta = np.int64(0)
    thresh = 0.5

    for cluster in range(num_clusters):
        cluster_center = cluster_df['centroid'][cluster]
        delta = abs(np.int64(daddr) - np.int64(cluster_center))

        cluster_min = cluster_center - cluster_width // 2
        cluster_max = cluster_center + cluster_width // 2

        if delta <= cluster_width_bytes:
            found_cluster = cluster
            found_delta = delta
            break

    if found_cluster == -1:
        return []
    else:
        delta_1h = np.zeros((1, n_delta_bits))
        delta_1h[0, found_delta + n_delta_bits // 2] = np.uint8(1)

        iaddr_ar = np.array([[iaddr]], dtype=np.uint64)

        predictions = []
        targets = models[cluster].predict([iaddr_ar, delta_1h])
        for t in targets:
            if t > thresh:
                predictions.append(t)

        return predictions

def init(clusters_file, arch_fname, weights_prefix):
    cluster_df = lr.import_clusters(clusters_file)
    num_clusters = len(cluster_df)
    models = lr.load_arch(num_clusters, arch_fname)
    lr.load_weights(models, num_clusters, weights_prefix)
    return cluster_df, models

if __name__ == "__main__":
    main()
