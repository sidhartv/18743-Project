import keras
import pandas as pd
import argparse
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K


def parse_args():
    parser = argparse.ArgumentParser(description="Run RNN")
    parser.add_argument("--input-file", type=str, help="Input Pandas DataFrame to parse")
    return parser.parse_args()

def create_network(n_categories, clusters):
    cluster_models = []
    for c in range(clusters):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(128, activation='sigmoid', input_shape=(1, n_categories+1)))
        model.add(keras.layers.Dense(n_categories, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        cluster_models.append(model)

    weight_ties_per_cluster = dict()
    for c in range(clusters):
        weight_assigns = []
        c_weights = cluster_models[c].trainable_weights
        for d in range(clusters):
            if d == c:
                continue
            d_weights = cluster_models[d].trainable_weights
            for cw, dw in zip(c_weights, d_weights):
                weight_assigns.append(tf.assign(dw, cw))

        weight_ties_per_cluster[c] = weight_assigns

    return cluster_models,weight_ties_per_cluster



def fit_network(models, in_data, in_cluster, out_data, weight_ties_per_cluster):
    losses = [[]]*(np.max(in_cluster) + 1)

    for i in range(len(in_data)):
        cluster = in_cluster[i][0]
        model = models[cluster]
        print("Iteration " + str(i) + ' cluster ' + str(cluster))
        h = model.fit(in_data[i].transpose().reshape(1,1,18), out_data[i].reshape(1,17), epochs=1)

        K.get_session().run(weight_ties_per_cluster[cluster])

        losses[cluster].append((i,h.history['loss']))




    plt.figure()
    for l in losses:
        plt.plot(*zip(*l))
    plt.ylabel('Training error')
    plt.xlabel('Epoch')
    plt.title('Training error over test trace')
    plt.show()
    return models



def parse_df(df):
    freqs = df['delta'].value_counts()
    freqs = freqs[freqs > 10]

    # Convert deltas to one-hot representation
    label_encoder = sklearn.preprocessing.LabelEncoder()
    deltas_labeled = label_encoder.fit_transform(df['delta'].values)
    onehot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
    deltas_onehot = onehot_encoder.fit_transform(deltas_labeled.reshape(-1,1))

    in_deltas = deltas_onehot[:-1]
    out_deltas = deltas_onehot[1:]

    in_iaddr = df['iaddr'].values[:-1].reshape(-1,1)
    in_cluster = df['cluster'].values[:-1].reshape(-1,1)
    in_data = np.concatenate((in_iaddr, in_deltas), axis=1)
    out_cluster = df['cluster'].values[1:].reshape(-1,1)
    out_data = out_deltas
    return (in_data, in_cluster, out_data, out_cluster)

def main():
    args = parse_args()

    df = pd.read_csv(args.input_file)
    in_data, in_clusters, out_data, out_clusters = parse_df(df)

    n_samples, n_input = in_data.shape

    div_pt = int(0.75 * n_samples)
    in_train = in_data[:div_pt]
    out_train = out_data[:div_pt]
    in_clusters_train = in_clusters[:div_pt]

    in_test = in_data[div_pt:]
    out_test = out_data[div_pt:]
    in_clusters_test = in_clusters[div_pt:]

    _,n_cats = in_test.shape

    n_clusters = max(np.max(in_clusters_train), np.max(in_clusters_test))
    model,weight_ties_per_cluster = create_network(n_cats-1, n_clusters)
    trained_model = fit_network(model, in_train, in_clusters_train, out_train, weight_ties_per_cluster)


if __name__ == '__main__':
    main()
