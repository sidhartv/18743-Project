import argparse
import sys
import numpy as np
import re
import logging
import pprint
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def init_args():
    parser = argparse.ArgumentParser(description="Determine miss clusters")
    parser.add_argument("dcache", type=argparse.FileType("r"),
                        help="Path to dcache.out of benchmark",
                        default="dcache.out")
    parser.add_argument("atrace", nargs="?", type=argparse.FileType("r"),
                        help="Path to atrace.out of benchmark",
                        default=sys.stdin)
    parser.add_argument("-o", "--outfile", nargs="?", type=argparse.FileType("w"),
                        help="Path to cluster.out of benchmark",
                        default=sys.stdout)

    parser.add_argument("-n", type=int, help="Number of clusters", default=8)
    parser.add_argument("-w", type=int, help="Maximum width of clusters (8-byte words)",
                        default=64)
    parser.add_argument("-t", type=int, help="Min. miss threhold", default=10)
    parser.add_argument("--pd-outfile", type=str)

    return parser.parse_args()

def parse_data_block(lines_view):
    data = {}

    for l in lines_view:
        match = re.match(data_re, l)
        if match:
            daddr, misses, hits = match.groups()
            data[daddr] = (misses, hits)

def parse_load_store(lines):
    """
    #
    # Detailed Stats
    # Instruction Address: 0x0000000000400b2c
    NumItems 0
    DATA:START
    #  counters
    # daddr          : dcache:miss        dcache:hit
    0x0000000000607078:           51          145
    DATA:END
    """

    iaddr_re = r"# Instruction Address:\s+(0x[a-f\d]+)"
    data_re = r"(0x[a-f\d]+):\s+(\d+)\s+(\d+)"

    start_idx = lines.index("# Detailed Stats\n")
    data_dict = {}
    while True:
        iaddr_offset = 1
        nitems_offset = 2
        data_offset = 6

        iaddr = re.match(iaddr_re, lines[start_idx+iaddr_offset]).group(1)
        if iaddr not in data_dict: data_dict[iaddr] = {}

        for l in lines[start_idx+data_offset:]:
            if l == "DATA:END\n": break

            daddr, misses, hits = re.match(data_re, l).groups()
            if daddr not in data_dict[iaddr]:
                data_dict[iaddr][daddr] = [int(misses), int(hits)]
            else:
                data_dict[iaddr][daddr][0] += misses
                data_dict[iaddr][daddr][1] += hits

        try:
            start_idx += 1 + lines[start_idx+1:].index("# Detailed Stats\n")
        except ValueError:
            break

    return data_dict

def prune_misses(stats, thresh):
    miss_data = {}

    for daddr, hits_misses in stats.items():
        _, misses = hits_misses
        if misses >= thresh:
            miss_data[daddr] = misses

    return miss_data

def parse_misses(infile, miss_thresh):
    logger = logging.getLogger()

    logger.info("Started loading cache data from file %s.", infile.name)
    lines = infile.readlines()
    logger.info("Finished loading cache data from file %s.", infile.name)

    start_idx = lines.index("# Begin LOAD/STORE stats\n")
    end_idx = lines.index("# End LOAD/STORE stats\n")
    cache_data = parse_load_store(lines[start_idx:end_idx])
    logger.info("Parsed cache miss data. %d unique instructions",
                len(cache_data))

    all_miss_data = {}

    for iaddr, stats in cache_data.items():
        miss_data = prune_misses(stats, miss_thresh)
        if len(miss_data) > 0: all_miss_data[iaddr] = miss_data

    n_data_addrs = sum(len(s) for _, s in all_miss_data.items())

    logger.info("Pruned cache miss data with min. threshold %d. %d unique " \
                "instructions remaining and %d (non)unique data addresses " \
                "remaining",
                miss_thresh, len(all_miss_data), n_data_addrs)

    return all_miss_data

def miss_samples(miss_data_dict):
    daddr2idx = {}
    idx = 0
    for iaddr, s in miss_data_dict.items():
        for daddr in s:
            if daddr not in daddr2idx:
                daddr2idx[daddr] = idx
                idx += 1

    n_addrs = len(daddr2idx)

    samples = np.zeros((n_addrs, 1), dtype=int)
    weights = np.zeros(n_addrs, dtype=int)

    for iaddr, s in miss_data_dict.items():
        for daddr, misses in s.items():
            idx = daddr2idx[daddr]
            samples[idx] = int(daddr, 16)
            weights[idx] += misses

    return samples, weights

def get_centroid(points):
    return np.sum(points) / points.shape[0]

def get_nearest_point(points, value):
    idx = np.abs(points - value).argmin()
    return points.flat[idx]

def miss_cluster(miss_samples, miss_weights, width, thresh):
    logger = logging.getLogger()

    db = DBSCAN(eps=width, min_samples=thresh)
    db.fit(miss_samples, sample_weight=miss_weights)

    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    logger.info("DBSCAN Silhouette Coefficient: %f",
                metrics.silhouette_score(miss_samples, labels))

    clusters = [miss_samples[labels == i] for i in range(n_clusters)]
    cluster_sizes = [len(c) for c in clusters]
    cluster_centroids = [0 for i in range(n_clusters)]

    for i, cluster in enumerate(clusters):
        centroid = get_centroid(cluster)
        closest_centroid = get_nearest_point(cluster, centroid)
        cluster_centroids[i] = closest_centroid

    return (np.array(clusters),
            np.array(cluster_sizes),
            np.array(cluster_centroids))

def align_addrs(addrs, align):
    assert(align > 0);
    assert(align & (align - 1) == 0);

    return np.bitwise_and(addrs, ~(align - 1))

def separate_clusters(raw_data, centroids, width):
    logger = logging.getLogger()
    cluster_inds = []
    for i, centroid in enumerate(centroids.flat):
        cluster_inds.append(np.abs(raw_data - centroid) < width)

        logger.info("Currently partitioning data for cluster %i "\
                    "with centroid 0x%x: %d addresses grouped",
                    i, centroid, np.count_nonzero(cluster_inds))

    return cluster_inds

def violin_misses(df):
    fig, (ax) = plt.subplots(1, 1)
    fig.suptitle("Clustering Density for Application")

    sns.violinplot(x="cluster", y="delta", hue="rw", data=df)

    return fig

def parse_accesses(infile, cluster_centroids, width, align):
    """
    0x7f9fbb809ff3: W 0x7fffcf848bb8
    """
    logger = logging.getLogger()

    data_re = r"(0x[a-f\d]+): ([RW]) (0x[a-f\d]+)"

    logger.info("Started loading access data from file %s.", infile.name)
    lines = infile.readlines()
    logger.info("Finished loading access data from file %s.", infile.name)

    raw_data = []
    idx = 0
    for l in lines:
        match = re.match(data_re, l)
        if match:
            iaddr, rw, daddr = match.groups()

            iaddr = int(iaddr, 16)
            daddr = int(daddr, 16) & ~(align - 1)
            cluster_idx = np.argmin(np.abs(cluster_centroids - daddr))
            delta = cluster_centroids[cluster_idx] - daddr
            if abs(delta) <= width:
                raw_data.append((iaddr, idx, rw, daddr, cluster_idx, delta))
                idx += 1

    return pd.DataFrame(raw_data, columns=("iaddr", "idx", "rw", "daddr",
                                           "cluster", "delta"))

def main():
    args = init_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
    logger = logging.getLogger()

    dcache_data_dict = parse_misses(args.dcache, args.t)
    samples, weights = miss_samples(dcache_data_dict)
    clusters, cluster_sizes, cluster_centroids = \
        miss_cluster(samples, weights, args.w * 8, args.t)
    print(cluster_centroids, cluster_sizes)

    # Align centroids to 64-byte (cache line) boundary
    cluster_centroids = align_addrs(cluster_centroids, 64)
    n_clusters = min(args.n, len(cluster_centroids))
    dense_inds = np.argpartition(cluster_sizes, -n_clusters)[-n_clusters:]

    logger.info("Calculated %d most dense clusters using DBSCAN", args.n)
    for idx in dense_inds:
        dense_cluster = clusters[idx]
        dense_cluster_size = cluster_sizes[idx]
        dense_cluster_centroid = cluster_centroids[idx]

        logger.info("Found cluster @0x%x with size %d", dense_cluster_centroid,
                    dense_cluster_size)

    dense_cluster_centroids = cluster_centroids[dense_inds]

    access_df = parse_accesses(args.atrace, dense_cluster_centroids, args.w * 8, 64)
    fig = violin_misses(access_df)
    plt.show()

    access_df.to_csv(args.pd_outfile)

if __name__ == "__main__":
    main()
