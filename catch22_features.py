import numpy as np


def get_catch22_features(series):
    import catch22

    # dict of {'names': [list of 22 feature names], 'values': [list of 22 feature values]}
    features = catch22.catch22_all(series)

    return features


if __name__ == "__main__":
    n_samples = 1000
    series = np.arange(n_samples)

    features = get_catch22_features(series)

    for name, val in zip(features['names'], features['values']):
        print('{} = {}'.format(name, val))
