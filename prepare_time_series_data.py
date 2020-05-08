import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, re, os
from pathlib import Path
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import utility
import catch22_features
import multiprocessing  # Pool, cpu_count


params = dict()
# Note: total_num_samples in series = 262144

# Parameters for preparing data
params['data_dir'] = '../Datasets/structural_damage_detection/Dataset A'
params['window_size'] = 160  # (window_size)    # Don't change this, we will use non-overlapping windows this way

# Following params don't need to be changed
params['num_samples_in_pool'] = 100000  # Increasing this number will increase memory usage in pool, but will reduce the no. of pools (slightly faster)
params['stride'] = params['window_size']  # stride when preparing windows (decides window overlap)
params['windows_per_undamaged_series'] = np.inf  # Limit the # of windows to extract from series (set np.inf for no limit)
params['windows_per_damaged_series'] = np.inf
params['exp_id'] = 'data_prep_win' + str(params['window_size'])
params['results_dir'] = 'output'


def load_series(file, columns):
    logging.info("Loading data from file: {}, columns: {}".format(file, columns))
    dataset_df = pd.read_csv(file, header=None, skiprows=11, delim_whitespace=True, dtype=float, usecols=columns)
    return dataset_df


def extract_dataset(series, window_size, stride, label, windows_per_series):
    # # Following 3 lines use np.split, but it cannot accommodate a stride
    # # Both methods (np.split and loop) take ~ the same time
    # num_windows = min(windows_per_series, len(series) // window_size)
    # series = series.iloc[:window_size * num_windows]
    # windows = np.split(series, num_windows)

    X = []
    num_windows = 0
    for i in range(0, len(series) - window_size - 1, stride):
        window = series.iloc[i:(i + window_size)]  #
        X.append(window)

        num_windows += 1
        if num_windows >= windows_per_series:
            break

    X_data = np.array(X).reshape((num_windows, window_size, 1))    # Required data shape: (sample_size, timesteps, data_dim)

    # X_data = np.array(windows).reshape((num_windows, window_size, 1))    # Required data shape: (sample_size, timesteps, data_dim)
    y_data = np.full(shape=num_windows, fill_value=label)  # Required data shape: (sample_size)

    return X_data, y_data


def parse_files(directory):
    files = glob.glob(directory + '/*.TXT')

    files_list = [] # List of tuples (filename, damaged_series_num)

    for file in files:
        filename = Path(file).stem  # Without the extension (eg: zzzAD10)
        splits = re.split('(\d+)', filename)
        # print(filename)
        # print(splits)
        L = len(splits)
        assert L == 1 or L == 3 # 1 for Undamaged case, 3 for Damaged case

        if L == 3:
            string = splits[0]
            damaged_series_num = int(splits[1])
            assert string[-1] == 'D'
            files_list.append((file, damaged_series_num))
        elif L == 1:    # Ignore the all Undamaged case for now
            pass

    return files_list


def normalize_series(series):
    scaler_obj = StandardScaler()
    scaled_series = scaler_obj.fit_transform(np.array(series).reshape((len(series), 1)))
    scaled_series = pd.Series(scaled_series.flatten())
    return scaled_series


def prepare_dataset(files, window_size, stride, win_per_damaged, win_per_undamaged):
    X_list, y_list = [], [] # lists of numpy arrays
    for (file, damaged_series_num) in files:
        logging.info('Processing file: {}, damaged_series_num: {}'.format(file, damaged_series_num))
        data_df = load_series(file, range(1, 31))   # 30 columns (col 0 is timestamp. col 1 to 30 are timeseries)
        for col in range(1, 31):
            series = data_df.iloc[:, col-1]

            # series = normalize_series(series)

            limit = win_per_damaged  # Limit for Undamaged series
            label = 'U' # Undamaged by default
            if col == damaged_series_num:
                # print('col {} is damaged'.format(col))
                label = 'D'
                limit = win_per_undamaged

            X_data, y_data = extract_dataset(series, window_size, stride, label, limit)
            X_list.append(X_data)
            y_list.append(y_data)
            # print('X: {}, y: {}'.format(X_data.shape, y_data.shape))

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)

    return X_all, y_all


def encode_labels(y_all):
    y_encoded = np.zeros(len(y_all))

    y_encoded[y_all == 'U'] = 0
    y_encoded[y_all == 'D'] = 1
    y_encoded[y_all == 'AD'] = 2  # For 3-class problem

    num_classes = len(np.unique(y_encoded))
    if num_classes > 2:  # Multiclass problem
        y_encoded = keras.utils.to_categorical(y_encoded, num_classes)

    return y_encoded


def prepare_and_save_dataset(params):
    files = parse_files(params['data_dir'])
    if len(files) == 0:
        logging.error("No raw data files found in {}".format(params['data_dir']))
        exit(1)
    # print(files)
    X_all, y_all = prepare_dataset(files, params['window_size'], params['stride'],
                                   params['windows_per_undamaged_series'], params['windows_per_damaged_series'])
    y_all = encode_labels(y_all)

    fname_suffix = utility.get_filename_suffix(params)

    X_filename = params['data_dir'] + '/X_all' + fname_suffix
    y_filename = params['data_dir'] + '/y_all' + fname_suffix
    np.save(X_filename, X_all)
    np.save(y_filename, y_all)
    logging.info('Saved datasets')

    return X_all, y_all


# Serial version
def convert_to_catch22(X):
    feature_rows = []
    for X_series in X:
        features_lists = catch22_features.get_catch22_features(X_series.flatten())
        features_dict = dict(zip(features_lists['names'], features_lists['values']))
        feature_rows.append(features_dict)

    X_all_catch22 = pd.DataFrame(feature_rows)
    # print('Finished conversion: num_samples = {}'.format(len(feature_rows)))
    return X_all_catch22


# Run multiple pools to control memory usage at any given point in time
# When a pool is finished, OS will clean-up memory in each worker process
def convert_to_catch22_pool(X_all, num_samples_in_pool):
    n_cores = max(1, multiprocessing.cpu_count() - 2)  # Leave 2 cpus for other work
    num_splits = X_all.shape[0] // num_samples_in_pool
    df_splits = np.array_split(X_all, num_splits)   # Initial split (across pools) to control memory usage at any time (in a pool)
    logging.info('num_splits = {}. samples_in_split = {}'.format(len(df_splits), len(df_splits[0])))

    converted_dfs = []
    for i, split in enumerate(df_splits):
        logging.info('Running pool {}/{}'.format(i, num_splits))
        df_pool_split = np.array_split(split, n_cores)  # Inner split among pool workers

        pool = multiprocessing.Pool(n_cores)
        df = pd.concat(pool.map(convert_to_catch22, df_pool_split)) # Map returns in order of the data split
        pool.close()
        pool.join()

        converted_dfs.append(df)
        # print('Finished pool #{}'.format(i))
        # When a pool is finished, OS will clean-up memory in each worker process

    final_converted_df = pd.concat(converted_dfs)
    logging.info('original_samples = {}, converted_samples = {}'.format(X_all.shape[0], final_converted_df.shape[0]))
    return final_converted_df


def convert_to_catch22_and_save_dataset(X_all, y_all, num_samples_in_pool):
    logging.info('Converting windows of time series to catch_22 feature dataset')

    X_all_catch22 = convert_to_catch22_pool(X_all, num_samples_in_pool)
    # print(X_all_catch22.head())

    fname_suffix = utility.get_filename_suffix(params)
    fname_suffix += '_catch22'

    X_filename = params['data_dir'] + '/X_all' + fname_suffix
    y_filename = params['data_dir'] + '/y_all' + fname_suffix
    np.save(X_filename, X_all_catch22)
    np.save(y_filename, y_all)
    logging.info('Saved catch22 datasets')

    return X_all_catch22, y_all


def main():
    utility.initialize(params)

    # ---------------------------------------------------
    # Dataset preparation

    # Extract windows from time series and prepare a dataset (and save it)
    # X_all, y_all = prepare_and_save_dataset(params)

    # Convert above prepared dataset to catch_22 features (and save it)
    X_all, y_all = utility.load_prepared_dataset(params)
    convert_to_catch22_and_save_dataset(X_all, y_all, params['num_samples_in_pool'])


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU
    main()
