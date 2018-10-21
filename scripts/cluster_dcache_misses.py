import argparse
import sys
import numpy as np
import re
import logging
import pprint
from sklearn.cluster import DBSCAN
from sklearn import metrics

def init_args():
    parser = argparse.ArgumentParser(description="Determine miss clusters")
    parser.add_argument("infile", nargs="?", type=argparse.FileType("r"),
                        help="Path to dcache.out of benchmark",
                        default=sys.stdin)
    parser.add_argument("outfile", nargs="?", type=argparse.FileType("r"),
                        help="Path to cluster.out of benchmark",
                        default=sys.stdout)

    parser.add_argument("-n", type=int, help="Number of clusters", default=8)
    parser.add_argument("-w", type=int, help="Maximum width of clusters (8-byte words)",
                        default=64)
    parser.add_argument("-t", type=int, help="Min. miss threhold", default=10)

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
    data = {}
    while True:
        iaddr_offset = 1
        nitems_offset = 2
        data_offset = 6

        iaddr = re.match(iaddr_re, lines[start_idx+iaddr_offset]).group(1)
        if iaddr not in data: data[iaddr] = {}

        for l in lines[start_idx+data_offset:]:
            if l == "DATA:END\n": break

            daddr, misses, hits = re.match(data_re, l).groups()
            if daddr not in data[iaddr]: data[iaddr][daddr] = [int(misses),
                                                               int(hits)]
            else:
                data[iaddr][daddr][0] += misses
                data[iaddr][daddr][1] += hits

        try:
            start_idx += 1 + lines[start_idx+1:].index("# Detailed Stats\n")
        except ValueError:
            break

    return data

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

def miss_samples(miss_data):
    daddr2idx = {}
    idx = 0
    for iaddr, s in miss_data.items():
        for daddr in s:
            if daddr not in daddr2idx:
                daddr2idx[daddr] = idx
                idx += 1

    n_addrs = len(daddr2idx)

    samples = np.zeros((n_addrs, 1), dtype=int)
    weights = np.zeros(n_addrs, dtype=int)

    for iaddr, s in miss_data.items():
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

    return clusters, cluster_sizes, cluster_centroids

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

def main():
    args = init_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
    logger = logging.getLogger()

    all_data = parse_misses(args.infile, args.t)
    samples, weights = miss_samples(all_data)
    clusters, cluster_sizes, cluster_centroids = \
        miss_cluster(samples, weights, args.w * 8, args.t)

    # Align centroids to 64-byte (cache line) boundary
    cluster_centroids = align_addrs(cluster_centroids, 64)
    dense_inds = np.argpartition(cluster_sizes, -args.n)[-args.n:]

    logger.info("Calculated %d most dense clusters using DBSCAN", args.n)
    for idx in dense_inds:
        dense_cluster = clusters[idx]
        dense_cluster_size = cluster_sizes[idx]
        dense_cluster_centroid = cluster_centroids[idx]

        logger.info("Found cluster @0x%x with size %d", dense_cluster_centroid,
                    dense_cluster_size)

    separated_data = separate_clusters(samples, cluster_centroids[dense_inds], args.w)

if __name__ == "__main__":
    main()
