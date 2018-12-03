from __future__ import print_function
from __future__ import division

import logging
import re

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN

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

def parse_accesses(infile, cluster_centroids, width, align):
    """
    0x7f9fbb809ff3 0x7fffcf848bb8 0
    (void *) (void *) bool <padding>
    """
    logger = logging.getLogger()

    fmt = np.dtype([("iaddr", np.uint64),
                    ("daddr", np.uint64),
                    ("is_read", np.bool_)], align=True)
    fmt_size = fmt.itemsize
    buf = infile.read()
    data_bytes = len(buf)
    logger.info("Struct fmt(%s) has size %d bytes, file is %d bytes.",
                str(fmt), fmt_size, data_bytes)

    assert(data_bytes % fmt_size == 0)

    logger.info("Started loading access data from file %s.", infile.name)
    raw_data = np.fromstring(buf, dtype=fmt)  # Creates a mutable copy
    del buf  # Free up the buffered memory early since it is not used
    logger.info("Finished loading access data from file %s -> %d elems.",
                infile.name, len(raw_data))

    raw_data["iaddr"] = align_addrs(raw_data["iaddr"], np.uint64(align))
    raw_data["daddr"] = align_addrs(raw_data["daddr"], np.uint64(align))

    logger.info("Generate RW mapping")
    raw_rw = np.full(raw_data.shape[0], "W", dtype=str)
    raw_rw[raw_data["is_read"]] = "R"

    raw_clusters = np.full(raw_data.shape[0], -1, dtype=int)
    raw_deltas = np.zeros(raw_data.shape[0], dtype=np.uint64)
    cluster_mask = np.full(raw_data.shape[0], False, dtype=np.bool_)
    for i, centroid in enumerate(cluster_centroids):
        cand_notset = np.logical_not(cluster_mask)
        logger.info("Evaluating %d candidates for cluster %d with centroid "
                    "0x%x...", np.count_nonzero(cand_notset), i, centroid)

        deltas = np.abs(raw_data["daddr"] - centroid)
        cand_matches = (deltas < np.uint64(width))
        new_matches = cand_matches & cand_notset

        logger.info("... found %d/%d matches.",
                    np.count_nonzero(new_matches), len(raw_data))

        raw_clusters[new_matches] = i
        raw_deltas[new_matches] = deltas[new_matches]
        cluster_mask[new_matches] = True

    filter_data = raw_data[cluster_mask]
    filter_rw = raw_rw[cluster_mask]
    filter_clusters = raw_clusters[cluster_mask]
    filter_deltas = raw_deltas[cluster_mask]

    return pd.DataFrame({"iaddr": filter_data["iaddr"],
                         "rw": filter_rw,
                         "daddr": filter_data["daddr"],
                         "cluster": filter_clusters,
                         "delta": filter_deltas})

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
    logger.info("Parsed cache miss data -> %d unique instructions",
                len(cache_data))

    all_miss_data = {}

    for iaddr, stats in cache_data.items():
        miss_data = prune_misses(stats, miss_thresh)
        if len(miss_data) > 0: all_miss_data[iaddr] = miss_data

    n_data_addrs = sum(len(s) for _, s in all_miss_data.items())

    logger.info("Pruned cache miss data with min. threshold %d.\n" \
                "\t> %d unique instructions remaining\n" \
                "\t> %d (non)unique data addresses remaining",
                miss_thresh, len(all_miss_data), n_data_addrs)

    return all_miss_data

def miss_samples(miss_data_dict):
    logger = logging.getLogger()

    daddr2idx = {}
    idx = 0
    for iaddr, s in miss_data_dict.items():
        for daddr in s:
            if daddr not in daddr2idx:
                daddr2idx[daddr] = idx
                idx += 1

    n_addrs = len(daddr2idx)

    samples = np.zeros((n_addrs, 1), dtype=np.uint64)
    weights = np.zeros(n_addrs, dtype=np.uint64)

    for iaddr, s in miss_data_dict.items():
        for daddr, misses in s.items():
            idx = daddr2idx[daddr]
            samples[idx] = int(daddr, 16)
            weights[idx] += misses

    logger.info("Converted dict version of miss_data to numpy arrays")

    return samples, weights

def get_centroid(points):
    return np.sum(points) / points.shape[0]

def get_nearest_point(points, value):
    idx = np.abs(points - value).argmin()
    return points.flat[idx]

def miss_cluster(miss_samples, miss_weights, width, thresh):
    logger = logging.getLogger()

    logger.info("Starting clustering with miss_samples(%s) width(%d) " \
                "thresh(%d)",
                str(miss_samples.shape), width, thresh)

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
            np.array(cluster_centroids, dtype=np.uint64))

def align_addrs(addrs, align):
    assert(align > 0);
    assert(align & (align - align.dtype.type(1)) == 0);

    return np.bitwise_and(addrs, ~(align - align.dtype.type(1)))

def separate_clusters(raw_data, centroids, width):
    logger = logging.getLogger()
    cluster_inds = []
    for i, centroid in enumerate(centroids.flat):
        cluster_inds.append(np.abs(raw_data - centroid) < width)

        logger.info("Currently partitioning data for cluster %d with " \
                    "centroid 0x%x\n"\
                    "\t> %d addresses grouped",
                    i, centroid, np.count_nonzero(cluster_inds))

    return cluster_inds
