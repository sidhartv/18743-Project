import itertools
import logging
import os.path
import sys

import keras
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sklearn.preprocessing
import sklearn.metrics
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
    parser.add_argument("--test-data", type=str, default="test_data.csv",
                        help="Input Pandas DataFrame to parse for data (CSV)")
    parser.add_argument("--cluster-data", type=str, default="cluster.csv",
                        help="Input Pandas DataFrame to parse for clusters (CSV)")

    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--weights-dir", type=str, default="weights/",
                        help="Output weights file (HDF5)")
    parser.add_argument("--arch-file", type=str, default="model.json",
                        help="Output architecture file (JSON)")

    parser.add_argument("--plot-history", type=str,
                        help="Path to training/valid. history plots (PNG)")

    parser.add_argument("--plot-confusion", type=str,
                        help="Path to confusion plot (PNG)")

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
        deltas_stacked = clus_in_data[:, :, 2:].reshape(times,
                                                        unroll,
                                                        n_delta_bits)

        in_data_stacked[cluster] = [clusters_stacked, iaddrs_stacked,
                                    deltas_stacked]
        out_data_stacked[cluster] = clus_out_data.reshape(times, n_delta_bits)

    trained_models, histories = lr.fit_network(models, in_data_stacked,
                                               out_data_stacked)

    if not args.no_save and args.weights_dir:
        lr.save_weights(trained_models, args.weights_dir,
                        weights_prefix="weights")

    if not args.no_save and args.arch_file:
        lr.save_arch(trained_models, args.arch_file)

    if args.plot_history:
        for c, history in enumerate(histories):
            fname = "c{}_".format(c) + args.plot_history
            title = "Cluster {} C-RNN Metrics".format(c)
            fig = plot_history(history, title=title)
            fig.savefig(fname)

    if args.plot_confusion:
        for c, model in enumerate(models):
            cluster_data, iaddr_data, delta_data = in_data_stacked[cluster]
            out_data = out_data_stacked[cluster]

            new_n_samples = min(10000, cluster_data.shape[0])
            sampled = lr.get_samples(cluster_data, iaddr_data, delta_data,
                                     out_data, new_n_samples=new_n_samples)
            cluster_data, iaddr_data, delta_data, y_true = sampled
            in_data = [cluster_data, iaddr_data, delta_data]

            y_pred = model.predict(in_data)
            y_true = lr.decode_deltas(y_true)
            y_pred = lr.decode_deltas(y_pred)

            classes = lr.decode_deltas(np.eye(n_delta_bits, n_delta_bits))
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=classes)

            fname = "c{}_".format(c) + args.plot_confusion
            title = "Cluster {} C-RNN Confusion Matrix".format(c)
            fig = plot_confusion(cm, classes, normalize=True, title=title)
            fig.savefig(fname)

def plot_confusion(cm, classes, normalize=False, title="Confusion"):
    if normalize:
        row_sums = np.sum(cm, axis=1)
        row_sums[row_sums == 0] = 1
        cm = cm.astype("float") / row_sums[:, np.newaxis]

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = np.max(cm) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], "0.2f"), horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")
    ax.set_title(title)
    fig.colorbar(im)

    fig.tight_layout()
    return fig

def plot_history(history, title="Cluster Metrics"):
    loss_list = [s for s in history.history if "loss" in s and "val" not in s]
    val_loss_list = [s for s in history.history if "loss" in s and "val" in s]
    acc_list = [s for s in history.history if "acc" in s and "val" not in s]
    val_acc_list = [s for s in history.history if "acc" in s and "val" in s]

    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True)

    for l in loss_list:
        training_loss = "Training Loss ({:0.5f})".format(history.history[l][-1])
        ax1.plot(epochs, history.history[l], "b-", label=training_loss)

    for l in val_loss_list:
        valid_loss = "Validation Loss ({:0.5f})".format(history.history[l][-1])
        ax1.plot(epochs, history.history[l], "g--", label=valid_loss)

    ax1.set_ylabel("Categorical Cross-Entropy")
    ax1.set_ylim(bottom=0)
    ax1.set_title("Measured Loss")
    ax1.legend()

    for l in acc_list:
        training_acc = "Training Accuracy ({:0.5f})".format(history.history[l][-1])
        ax2.plot(epochs, history.history[l], "b-", label=training_acc)

    for l in val_acc_list:
        valid_acc = "Validation Accuracy ({:0.5f})".format(history.history[l][-1])
        ax2.plot(epochs, history.history[l], "g--", label=valid_acc)

    ax2.set_ylabel("Prediction Accuracy")
    ax2.set_ylim(bottom=0., top=1.)
    ax2.set_title("Measured Accuracy")
    ax2.legend()

    ax2.set_xlabel("Epochs")
    fig.suptitle(title)

    return fig

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
    thresh = 0.5

    deltas = np.int64(daddr) - cluster_df["centroid"].values.astype(np.int64);
    abs_deltas = np.abs(deltas)

    logger.info("Deltas for iaddr(%0x) daddr(%0x) are:\n" \
                "\t> %s\n" \
                "ANY?(%s)\n",
                iaddr, daddr, np.array_str(deltas),
                str(np.any(abs_deltas <= max_cluster_width_bytes)))

    return np.any(abs_deltas <= max_cluster_width_bytes)

infer_reqs = {}
def infer(iaddr, daddr_ual, is_read, rnn_handle):
    cluster_df, models = rnn_handle
    daddr = np.uint64(daddr_ual) & ~np.uint64(blocksize - 1)

    thresh = 0.5
    unroll = rnn_handle[1][0].input_shape[0][1]

    deltas = np.int64(daddr) - cluster_df["centroid"].values.astype(np.int64);
    found_cluster = np.argmin(np.abs(deltas))
    found_delta = deltas[found_cluster]

    if np.abs(found_delta) > max_cluster_width_bytes:
        logging.warning("Attempted to perform inference on invalid access " \
                        "for cluster(%d): iaddr(%0x) daddr(%0d) is_read(%d).",
                        found_cluster, iaddr, daddr, is_read)

        return []
    else:
        if found_cluster not in infer_reqs:
            new_iaddrs = np.zeros((unroll, 1), dtype=np.uint64)
            new_deltas = np.zeros((unroll, n_delta_bits), dtype=np.uint8)
            infer_reqs[found_cluster] = [new_iaddrs, new_deltas]

            logging.info("Allocated new history stream for cluster(%d) with "\
                         "iaddr(%0x) daddr(%0x)",
                         found_cluster, iaddr, daddr)

        delta_1h = lr.encode_deltas(np.array([found_delta]))

        prev_iaddrs, prev_deltas = infer_reqs[found_cluster]
        infer_reqs[found_cluster][0] = np.roll(prev_iaddrs, 1, axis=0)
        infer_reqs[found_cluster][1] = np.roll(prev_deltas, 1, axis=0)
        infer_reqs[found_cluster][0][-1] = iaddr
        infer_reqs[found_cluster][1][-1] = delta_1h

        if infer_reqs[found_cluster][0][0] == np.uint64(0):
            logging.info("Still in the process of building history for" \
                         "cluster(%d): %d/%d placeholders remaining.",
                         found_cluster,
                         np.count_nonzero(infer_reqs[found_cluster][0]),
                         unroll)

            return []

        clusters_pred = np.full((unroll, 1), found_cluster, dtype=np.uint8)
        iaddrs_pred, deltas_pred = infer_reqs[found_cluster]

        pred_data = [clusters_pred.reshape(1, unroll, -1),
                     iaddrs_pred.reshape(1, unroll, -1),
                     deltas_pred.reshape(1, unroll, -1)]

        predictions = lr.predict(models, cluster_df, found_cluster, pred_data)

        logger.info("*NEW prefetch predictions for cluster(%d) len(%d):\n" \
                    "*\t> %s.",
                    found_cluster, len(predictions), str(predictions))

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
