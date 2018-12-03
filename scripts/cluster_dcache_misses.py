#  from __future__ import print_function
#  from __future__ import division

import libcluster as lc
import argparse
import gzip
import logging
import re
import sys
import tarfile

#  import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#  import seaborn as sns
from sklearn import metrics
from sklearn.cluster import DBSCAN

# x86-64 cache block size is 512-bytes and word size is 8-bytes
blocksize = np.uint64(512)
wordsize = np.uint64(8)

def init_args():
    parser = argparse.ArgumentParser(description="Determine miss clusters")
    parser.add_argument("datatar", nargs="?", type=argparse.FileType("rb"),
                        help="Path to data.tar(.gz)[dcache.out, atrace.out]")
    parser.add_argument("--dcache", type=argparse.FileType("r"),
                        help="Path to dcache.out of benchmark")
    parser.add_argument("--atrace", type=argparse.FileType("rb"),
                        help="Path to atrace.out of benchmark")

    parser.add_argument("-n", type=int, default=8, help="Number of clusters")
    parser.add_argument("-w", type=int, default=8,
                        help="Maximum width of clusters in cache blocks")
    parser.add_argument("-t", type=int, default=10, help="Min. miss threhold")

    parser.add_argument("--logfile", type=str, help="Path to logfile")
    parser.add_argument("-o", "--outfile", type=str,
                        help="Path to output file for CSV-formatted clusters")
    parser.add_argument("--test-outfile", type=str,
                        help="Path to output file for CSV-formatted test data")
    parser.add_argument("--violin-outfile", type=str,
                        help="Path to output file for Violin plots")

    args = parser.parse_args()

    if args.datatar:
        print("Reading in data from archive({}).".format(args.datatar))

        try:
            gzfile = gzip.GzipFile(fileobj=args.datatar)
            tar = tarfile.TarFile(fileobj=gzfile)
        except:
            print("Archive({}) was not gzipped... attempting to open " \
                  "uncompressed file".format(args.datatar))

            tar = tarfile.TarFile(fileobj=args.datatar)

        members = tar.getmembers()
        names = tar.getnames()
        print("\t> Tarfile contents: {}".format(names))
        if ("dcache.out" in names) and ("atrace.out" in names):
            dcache_idx = names.index("dcache.out")
            atrace_idx = names.index("atrace.out")

            args.dcache = tar.extractfile(members[dcache_idx])
            args.atrace = tar.extractfile(members[atrace_idx])
        else:
            print("ERR> Unable to locate dcache.out and atrace.out in archive.")
            sys.exit(1)

    elif args.dcache and args.atrace:
        print("Reading in data from dcache({}) and atrace({}).".format(
            args.dcache.name, args.atrace.name))
    else:
        print("Must specify either [datatar] or --dcache AND --atrace.")
        sys.exit(1)

    return args

def violin_misses(df):
    fig, (ax) = plt.subplots(1, 1)
    fig.suptitle("Clustering Density for Application")

    sns.violinplot(x="cluster", y="delta", hue="rw", data=df, inner="quartiles",
                   split=True, ax=ax)

    ax.set_xlabel("Cluster #")
    ax.set_ylabel("Delta (bytes)")
    ax.legend(title="Read/Write")

    return fig

def main():
    args = init_args()
    if args.logfile:
        logging.basicConfig(format="%(levelname)s: %(message)s",
                            level=logging.DEBUG, filename=args.logfile)
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s",
                            level=logging.DEBUG)

    logger = logging.getLogger()

    dcache_data_dict = lc.parse_misses(args.dcache, args.t)
    samples, weights = lc.miss_samples(dcache_data_dict)
    clusters, cluster_sizes, cluster_centroids = \
        lc.miss_cluster(samples, weights, args.w * blocksize, args.t)

    # Align centroids to 64-byte (cache line) boundary
    cluster_centroids = lc.align_addrs(cluster_centroids, blocksize)
    n_clusters = min(args.n, len(cluster_centroids))
    dense_inds = np.argpartition(cluster_sizes, -n_clusters)[-n_clusters:]

    logger.info("Calculated %d most dense clusters using DBSCAN.", args.n)
    for idx in dense_inds:
        dense_cluster = clusters[idx]
        dense_cluster_size = cluster_sizes[idx]
        dense_cluster_centroid = cluster_centroids[idx]

        logger.info("Found cluster @0x%x with size %d",
                    dense_cluster_centroid, dense_cluster_size)

    dense_cluster_centroids = cluster_centroids[dense_inds]

    access_df = lc.parse_accesses(args.atrace, dense_cluster_centroids,
                                  args.w * blocksize, blocksize)

    if args.outfile:
        cluster_df = pd.DataFrame({
            "centroid": cluster_centroids[dense_inds],
            "size": cluster_sizes[dense_inds]
        })

        cluster_df.to_csv("cluster.csv")

    '''
    if args.violin_outfile:
        fig = violin_misses(access_df)
        fig.savefig(args.violin_outfile)
    '''

    if args.test_outfile:
        access_df.to_csv("test_data.csv")

if __name__ == "__main__":
    main()
