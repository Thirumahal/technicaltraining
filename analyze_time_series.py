import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility
# from statsmodels.graphics.tsaplots import plot_acf

params = dict()

# Parameters for initial analysis (visualizing etc)
params['data_file'] = '../Datasets/structural_damage_detection/Dataset A/zzzAD3.TXT'
params['sampling_rate'] = 1024  # Hz
params['plot_length'] = 2000
params['results_dir'] = 'output'


def load_series(file, columns):
    logging.info("Loading data from file: {}, columns: {}".format(file, columns))
    dataset_df = pd.read_csv(file, header=None, skiprows=11, delim_whitespace=True, dtype=float, usecols=columns)
    return dataset_df


def plot(x, y, xlabel=None, y_label=None, title=None, new_figure=True, **kwargs):
    if new_figure:
        fig = plt.figure()
        utility.add_figure_to_save(fig, title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(y_label)
    plt.plot(x, y, **kwargs)
    plt.legend()


def plot_fourier_transform(series, sampling_rate, title_suffix=''):
    signal_size = len(series)
    fft_series = np.abs(np.fft.fft(series))
    # freq = np.fft.fftfreq(signal_size, sampling_interval)
    # plot(freq, fft_series, xlabel='w (cycles/sec)', y_label='DFT amplitude', title='Frequency spectrum (DFT): zzzAD3 - series #1')
    freq_manual = np.linspace(0, sampling_rate, signal_size)
    plot(freq_manual[:signal_size // 2], fft_series[:signal_size // 2], xlabel='Freq (Hz)', y_label='DFT amplitude'
         , title='Frequency spectrum (DFT): ' + title_suffix, c='orange', alpha=0.5)


def visualize_series(filename, columns, suffix):
    data_df = load_series(filename, columns)

    plot_len = params['plot_length']
    full_series = data_df.iloc[:, 0]
    plot_series = data_df.iloc[0:plot_len, 0]

    plot(range(len(plot_series)), plot_series, xlabel='n', y_label='y', title=suffix)

    # norm_series = normalize_series(full_series)
    # plot(range(len(full_series)), full_series, xlabel='n', y_label='y', title=suffix)
    # plot(range(len(norm_series)), norm_series, xlabel='n', y_label='y', label='Normalized', new_figure=False, alpha=0.4)

    plot_fourier_transform(full_series, params['sampling_rate'], suffix + " (full series)")

    return full_series


def visualize_undersampling(series, suffix):
    plot_len = params['plot_length']
    for step in [2, 4, 16, 32, 64]:
        us_series = series.iloc[::step]
        plot_series = us_series.iloc[0:plot_len//step]
        plot(range(len(plot_series)), plot_series, xlabel='n', y_label='y', title= suffix + ': undersampled step = {}'.format(step))
        # plot_fourier_transform(us_series, params['sampling_rate']/step, "Undersampled series (gap = {})".format(step))


def visualize_truncation(series, suffix):
    # for cut_factor in [2, 4, 8, 16]:
    #     series = full_series.iloc[0:len(full_series) // cut_factor]
    #     plot_fourier_transform(series, params['sampling_rate'], "Cut series (cut_factor = {})".format(cut_factor))

    for size in [400, 200, 160, 120, 100, 80, 40]:   # 400, 200, 160, give frequency plots very similar to the original
        small_series = series.iloc[0:size]
        plot_fourier_transform(small_series, params['sampling_rate'], suffix + " truncated: series_size = {}".format(size))


def main():
    utility.initialize(params)

    # ---------------------------------------------------
    # Initial analysis (visualization)

    series_1_u = visualize_series(params['data_file'], [1], "Undamaged (s#1)")
    print(series_1_u.head())

    series_3_d = visualize_series(params['data_file'], [3], "Damaged (s#3)")

    visualize_undersampling(series_1_u, "Undamaged (s#1)")

    visualize_truncation(series_1_u, "Undamaged (s#1)")

    # cut_series = full_series.iloc[0:len(full_series) // 16]
    # plot_acf(cut_series, lags=np.arange(-500, 500))

    utility.save_all_figures(params['results_dir'])


if __name__ == "__main__":
    main()
