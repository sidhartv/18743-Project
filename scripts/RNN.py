import logging

import keras
import pandas as pd
import argparse
import sklearn.preprocessing
import numpy as np

import librnn as lr

# x86-64 cache block size is 512-bytes and word size is 8-bytes
blocksize = np.uint64(512)
wordsize = np.uint64(8)

# Maximum width of clusters in cache blocks (inc 0)
n_delta_bits = 8 + 1

def parse_args():
    parser = argparse.ArgumentParser(description="Run RNN")
    parser.add_argument("--test_data", type=str, default="test_data.csv",
                        help="Input Pandas DataFrame to parse for data (CSV format)")
    parser.add_argument("--cluster_data", type=str, default="cluster.csv",
                        help="Input Pandas DataFrame to parse for clusters (CSV format)")

    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--weights_file", type=str, default="weights.h5",
                        help="Output weights file (HDF5 format)")
    parser.add_argument("--arch_file", type=str, default="model.json",
                        help="Output architecture file (JSON format)")

    return parser.parse_args()

def create_and_train():
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)

    args = parse_args()
    logger = logging.getLogger()

    test_df = lr.import_tests(args.test_data)
    cluster_df = lr.import_clusters(args.cluster_data)
    logger.info("Finished loading test data and cluster information.")
    in_data, out_data = lr.parse_tests(test_df, cluster_df, n_delta_bits)

    n_clusters = len(cluster_df)
    n_dct_iaddrs = test_df["iaddr"].nunique()

    in_data_stacked = np.vstack(v for _, v in sorted(in_data.items()))
    out_data_stacked = np.vstack(v for _, v in sorted(out_data.items()))

    models,weight_tie = lr.create_network(n_clusters, n_dct_iaddrs, 4, n_delta_bits)

    times, features = in_data_stacked.shape
    cluster_inputs = in_data_stacked[:, 0:n_clusters].reshape(times, n_clusters)

    cluster_inputs = np.argmax(cluster_inputs, axis=1)
    iaddr_inputs = in_data_stacked[:, n_clusters].reshape(times, 1,
                                                              1)
    delta_inputs = in_data_stacked[:, n_clusters+1:].reshape(times, 1,
                                                                 n_delta_bits)

    trained_models = lr.fit_network(models,
            [iaddr_inputs, delta_inputs],
            cluster_inputs,
            out_data_stacked,
            weight_tie)

    if not args.no_save and args.weights_file:
        lr.save_weights(trained_models, args.weights_file)
        logging.info("Successfully saved weights to file" \
                     "{}".format(args.weights_file))
    if not args.no_save and args.arch_file:
        lr.save_arch(trained_models, args.arch_file)
        logging.info("Successfully saved architecture to file" \
                     "{}".format(args.arch_file))

def load_model(num_clusters, arch_fname="model.json", weights_prefix="weights"):
    model = lr.load_arch(num_clusters, arch_fname)
    lr.load_weights(model, num_clusters, weights_fname)

    return model

def main():
    pass

def infer(iaddr, delta):
    found_cluster = -1
    for cluster in range(num_clusters):
        cluster_center = cluster_df['centroid'][cluster]
        cluster_min = cluster_center - cluster_width // 2
        cluster_max = cluster_center + cluster_width // 2

        if iaddr >= cluster_min and iaddr <= cluster_max:
            found_cluster = cluster
            break

    if found_cluster == -1:
        return []
    else:
        predictions = []
        targets = models[cluster].predict([iaddr, daddr])
        for t in targets:
            if t > 0.5:
                predictions.append(t)

        return predictions




def init(clusters_file, arch_fname, weights_prefix):
    cluster_df = lr.import_clusters(clusters_file)
    num_clusters = len(cluster_df)
    models = lr.load_arch(num_clusters, arch_fname)
    lr.load_weights(models, num_clusters, weights_prefix)
    return cluster_df, models

if __name__ == "__main__":
    create_and_train()
