import keras
import pandas as pd
import argparse
import sklearn.preprocessing
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="Run RNN")
    parser.add_argument("--input-file", type=str, help="Input Pandas DataFrame to parse")
    return parser.parse_args()

def create_network(n_categories):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(n_categories-1, input_shape=(1,n_categories), activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



def fit_network(model, in_data, out_data):
    times,samples = in_data.shape
    in_data = in_data.reshape((times,1,samples))

    model.fit(in_data, out_data, epochs=100)
    return model



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
    in_data = np.concatenate((in_iaddr, in_cluster, in_deltas), axis=1)

    out_cluster = df['cluster'].values[1:].reshape(-1,1)
    out_data = np.concatenate((out_cluster, out_deltas), axis=1)
    return (in_data, out_data)

def main():
    args = parse_args()

    df = pd.read_csv(args.input_file)
    in_data, out_deltas = parse_df(df)

    _, n_input = in_data.shape

    model = create_network(n_input)
    trained_model = fit_network(model, in_data, out_deltas)

if __name__ == '__main__':
    main()
