import logging, math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sns
import utility

params = dict()
params['data_file'] = '../Datasets/structural_damage_detection/Damage_Results_A.csv'
params['results_dir'] = 'output'


def plot_class_distribution(y, dataset_name=''):
    fig = plt.figure()
    utility.add_figure_to_save(fig, 'class_dist_' + dataset_name)
    plt.title('Class distribution of dataset: ' + dataset_name)
    unique_counts = y.value_counts()
    labels = list(unique_counts.index)

    logging.info('Class distribution of dataset: ' + dataset_name + ': \n{}'.format(unique_counts))

    # count = unique_counts.columns[0]
    plt.bar(labels, unique_counts, width=0.5)

    for i in range(len(labels)):
        plt.annotate(str(unique_counts.iloc[i]), xy=(i, unique_counts.iloc[i] + 3))

    # n, bins, patches = plt.hist(unique_counts)
    plt.xlabel('Class')
    plt.ylabel('No. of samples')


def do_pca(X, num_components):
    n = num_components
    pca = PCA(n_components=n)
    # pca = KernelPCA(n_components=n, kernel='cosine')
    X_pca = pca.fit_transform(X)

    col_names = ["PCA_comp_" + str(i) for i in range(1, n+1)]
    X_pca_df = pd.DataFrame(X_pca, columns=col_names)

    return X_pca_df


def plot_dataset_2d(X, y):
    assert X.shape[1] == 2
    assert X.shape[0] == y.shape[0]

    labels = y.unique()

    plt.figure()
    plt.title("2D PCA component data distribution")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    for label in labels:
        X_label = X[y == label]
        plt.scatter(X_label[X_label.columns[0]], X_label[X_label.columns[1]])

    plt.legend(list(labels), loc='lower right')


def plot_correlation_matrix(X):
    logging.info('Plotting correlation matrix')
    plt.matshow(X.corr())
    plt.xticks(range(X.shape[1]), X.columns, fontsize=7, rotation=90)
    plt.yticks(range(X.shape[1]), X.columns, fontsize=7)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', y=-0.05)


def plot_feature_distributions_nxn_grid(X, y=None, n=3):
    grid_size = n*n
    num_features = X.shape[1]
    num_plots = math.ceil(num_features / grid_size)

    if y is not None:
        labels = y.unique()

    for i in range(num_plots):
        start = i*grid_size
        end = (i+1)*grid_size
        subset = pd.DataFrame(X, columns=X.columns[start: end])
        plt.figure(figsize=(10, 10))
        plt.title('Feature distributions: features {} to {}'.format(start, end-1))

        for j in range(subset.shape[1]):
            plt.subplot(n, n, j + 1)
            feature = subset[subset.columns[j]]

            if y is None:   # Plot the feature distribution (full)
                sns.distplot(feature)
            else:   # Plot feature distrbituions separated by class
                for label in labels:
                    feature_label = feature[y == label]
                    sns.distplot(feature_label, label=label, hist_kws={"alpha": 0.4})
                plt.legend()


if __name__ == "__main__":
    utility.setup_logging(params['results_dir'])

    data_df = utility.load_dataset(params['data_file'])

    X_all = data_df.drop(columns=['Series', 'Class'], errors='ignore')
    y_all = data_df['Class']

    plot_class_distribution(y_all)

    X_pca_2d = do_pca(X_all, num_components=2)

    plot_dataset_2d(X_pca_2d, y_all)

    plot_correlation_matrix(X_all)

    plot_feature_distributions_nxn_grid(X_all, n=3)

    plot_feature_distributions_nxn_grid(X_all, y_all, n=3)

    plt.show()

