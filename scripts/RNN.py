import logging

import keras
import pandas as pd
import argparse
import sklearn.preprocessing
import numpy as np

import librnn as lr

def parse_args():
    parser = argparse.ArgumentParser(description="Run RNN")
    parser.add_argument("--input-file", type=str, help="Input Pandas DataFrame to parse")
    return parser.parse_args()

def main():
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)

    args = parse_args()

    logger = logging.getLogger()

    test_df = lr.import_tests()
    cluster_df = lr.import_clusters()
    logger.info("Finished loading test data and cluster information.")
    in_data, out_data, n_delta_bits= lr.parse_tests(test_df)

    n_dct_clusters = test_df["cluster"].nunique()
    n_dct_iaddrs = test_df["iaddr"].nunique()

    in_data_stacked = np.vstack(v for _, v in sorted(in_data.items()))
    out_data_stacked = np.vstack(v for _, v in sorted(out_data.items()))

    model = lr.create_network(n_dct_clusters, n_dct_iaddrs, 4, n_delta_bits)

    times, features = in_data_stacked.shape
    cluster_inputs = in_data_stacked[:, :n_dct_clusters].reshape(times, 1,
                                                                 n_dct_clusters)
    iaddr_inputs = in_data_stacked[:, n_dct_clusters].reshape(times, 1,
                                                              1)
    delta_inputs = in_data_stacked[:, n_dct_clusters+1:].reshape(times, 1,
                                                                 n_delta_bits)

    trained_model = lr.fit_network(model, [cluster_inputs, iaddr_inputs,
                                           delta_inputs], [out_data_stacked])

if __name__ == '__main__':
    main()
