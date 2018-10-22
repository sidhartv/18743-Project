import keras
import pandas as pd
import argparse
import sklearn.preprocessing

def parse_args():
    parser = argparse.ArgumentParser(description="Run RNN")
    parser.add_argument("--num-neurons", type=int, help="Number of LSTM units to use", default=16)
    parser.add_argument("--input-file", type=str, help="Input Pandas DataFrame to parse")
    return parser.parse_args()

def create_network(n_neurons):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(n_neurons))
    model.compile()

    return model

def fit_network(model, in_data, out_data):
    model.fit(in_data, out_data)
    return model

def parse_df(df):
    freqs = df['delta'].value_counts()
    freqs = freqs[freqs > 10]

    # Convert deltas to one-hot representation
    enc = sklearn.preprocessing.OneHotEncoder(freqs.index.values, sparse=False, handle_unknown='ignore')

    in_deltas = enc.fit_transform(df['delta'].values[:-1].reshape(-1,1))
    out_deltas = enc.fit_transform(df['delta'].values[1:].reshape(-1,1))

    in_iaddr = df['iaddr'].values[:-1]
    in_data = np.concatenate(in_addr, in_deltas)

    return (in_data, out_deltas)

def main():
    args = parse_args()
    #model = create_network(args.num_neurons)

    df = pd.read_csv(args.input_file)
    in_data, out_deltas = parse_df(df)

    trained_model = fit_network(model, in_data, out_deltas)

if __name__ == '__main__':
    main()
