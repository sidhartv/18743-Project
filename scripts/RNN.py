import logging
import os.path
import sys

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

    logger.info("Started loading test data and cluster information.")
    test_df = lr.import_tests(args.test_data)
    cluster_df = lr.import_clusters(args.cluster_data)
    logger.info("Finished loading test data and cluster information.")
    in_data, out_data = lr.parse_tests(test_df, cluster_df)

    n_clusters = len(cluster_df)
    n_dct_iaddrs = test_df["iaddr"].nunique()

    cluster_emb_len = 4
    iaddr_emb_len = 4
    models,weight_tie = lr.create_network(n_clusters, cluster_emb_len,
                                          n_dct_iaddrs, iaddr_emb_len)

    in_data_stacked = {}
    out_data_stacked = {}
    for cluster in range(n_clusters):
        clus_in_data = in_data[cluster]
        clus_out_data = out_data[cluster]

        times, unroll, features = clus_in_data.shape
        out_times, out_features = clus_out_data.shape
        assert(times == out_times)

        clusters_stacked = clus_in_data[:, :, 0].reshape(times, unroll, 1)
        iaddrs_stacked = clus_in_data[:, :, 1].reshape(times, unroll, 1)
        deltas_stacked = clus_in_data[:, :, 2:].reshape(times, unroll, n_delta_bits)

        in_data_stacked[cluster] = [clusters_stacked, iaddrs_stacked,
                                    deltas_stacked]
        out_data_stacked[cluster] = clus_out_data.reshape(times, n_delta_bits)

    trained_models = lr.fit_network(models, in_data_stacked, out_data_stacked)

    if not args.no_save and args.weights_dir:
        lr.save_weights(trained_models, args.weights_dir,
                        weights_prefix="weights")

    if not args.no_save and args.arch_file:
        lr.save_arch(trained_models, args.arch_file)

def load_model(n_clusters, arch_fname="model.json", weights_prefix="weights"):
    model = lr.load_arch(n_clusters, arch_fname)
    lr.load_weights(model, n_clusters, weights_fname)

    return model

def main():
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)

    args = parse_args()
    create_and_train(args)

def inferchk(iaddr, daddr_ual, is_read, rnn_handle):
    logger = logging.getLogger()

    cluster_df, models = rnn_handle
    daddr = np.uint64(daddr_ual) & ~np.uint64(blocksize - 1)

    found_cluster = -1
    found_delta = np.int64(0)
    thresh = 0.5

    deltas = np.int64(daddr) - cluster_df["centroid"].values.astype(np.int64);
    abs_deltas = np.abs(deltas)

    logger.info("Deltas for iaddr(%0x) daddr(%0x) are:\n" \
                "\t> %s\n" \
                "ANY?(%s)\n",
                iaddr, daddr, np.array_str(deltas),
                str(np.any(abs_deltas <= max_cluster_width_bytes)))

    return np.any(abs_deltas <= max_cluster_width_bytes)

def infer(iaddr, daddr_ual, is_read, rnn_handle):
    cluster_df, models = rnn_handle
    daddr = np.uint64(daddr_ual) & ~np.uint64(blocksize - 1)

    found_cluster = -1
    found_delta = np.int64(0)
    thresh = 0.5

    for cluster in range(n_clusters):
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
        return [cluster_df["centroid"][found_cluster]]

        delta_1h = np.zeros((1, n_delta_bits))
        delta_1h[0, found_delta + n_delta_bits // 2] = np.uint8(1)

        iaddr_ar = np.array([[iaddr]], dtype=np.uint64)

        predictions = []
        targets = models[cluster].predict([iaddr_ar, delta_1h])
        for t in targets:
            if t > thresh:
                predictions.append(t)

        return predictions

def init(arch_epath, weights_edirpath, cluster_epath):
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG, filename="rnn.log")

    logger = logging.getLogger()
    logger.info("TESTING\n")

    logger.info("Embedded Python Environment:"\
                "\t> PATH(%s)\n" \
                "\t> version(%s)\n" \
                "\t> executable(%s)\n" \
                "\t> PREFIX(%s)\n",
                sys.path, sys.version, sys.executable, sys.prefix)

    arch_path = os.path.expandvars(arch_epath)
    weights_dirpath = os.path.expandvars(weights_edirpath)
    cluster_path = os.path.expandvars(cluster_epath)

    try:
        cluster_df = lr.import_clusters(cluster_path)
        n_clusters = len(cluster_df)
        logger.info("Successfully imported clusters (discovered %d)", n_clusters)
    except Exception as e:
        logger.info("File %s could not be opened properly -> %s.", cluster_path,
                    e.message)

        exit(1)

    try:
        n_models = n_clusters
        models = lr.load_arch(n_models, arch_path)
    except Exception as e:
        logger.info("File %s could not be opened properly -> %s.", arch_path,
                    e.message)

        exit(1)

    try:
        lr.load_weights(models, n_clusters, weights_dirpath, weights_prefix="weights")
    except Exception as e:
        logger.info("Files in directory %s could not be opened properly -> %s.",
                    weights_dirpath, e.message)

        exit(1)

    logger.info("Successfully initialized RNN")

    return (cluster_df, models)

def cleanup():
    pass

if __name__ == "__main__":
    main()
