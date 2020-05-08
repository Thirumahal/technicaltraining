import logging, time, os, sys
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import models.ann, models.rf, models.svm, models.lstm, models.lr, models.xgboost
import utility


def create_and_train_model(params, dataset):
    model_name = params['model']
    logging.info('Initializing model: {}'.format(model_name))

    if model_name == 'ann':
        model = models.ann.ANN()
    elif model_name == 'rf':
        model = models.rf.RandomForest()
    elif model_name == 'svm':
        model = models.svm.SVM()
    elif model_name == 'lr':
        model = models.lr.LogisticRegression()
    elif model_name == 'xgb':
        model = models.xgboost.XGBoost()
    elif model_name == 'lstm':
        model = models.lstm.LSTMClassifier()
    else:
        assert False

    model.initialize(params)

    (X_train, y_train), (X_val, y_val) = dataset

    logging.info('Training model')
    t0 = time.time()

    history = model.fit(X_train, y_train, X_val, y_val)

    time_to_train = time.time() - t0
    logging.info('Training complete. time_to_train = {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    return model, history


def scale_dataset(dataset):
    logging.info('Scaling train and test sets')
    (X_train, y_train), (X_test, y_test) = dataset
    X_train_scaled, scaler_obj = utility.scale_training_set(X_train)

    X_test_scaled = X_test
    if X_test is not None:
        X_test_scaled = scaler_obj.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled)

    # data_analysis.plot_feature_distributions_nxn_grid(X_train_scaled, n=3)
    # data_analysis.plot_class_distribution(y_train)

    dataset = (X_train_scaled, y_train), (X_test_scaled, y_test)

    return dataset


def evaluate_model(model, X, y_true, dataset_name):
    y_pred = model.predict_classes(X)  # Integer labels
    y_pred_prob = model.predict(X)

    if y_true.ndim == 1:
        num_classes = 2
    else:    # Multiclass (3-class) problem
        num_classes = y_true.shape[1]

    if num_classes > 2:
        y_true_one_hot = y_true
        y_true = y_true.argmax(axis=1)  # Integer labels

    utility.print_evaluation_report(y_true, y_pred, dataset_name)
    conf_mat_df = utility.plot_confusion_matrix(y_true, y_pred, dataset_name)

    logging.info('{} confusion matrix below\n{}'.format(dataset_name, conf_mat_df))

    metrics_dict = {'dataset': dataset_name}

    if num_classes == 2: # Binary (2-class) problem
        metrics_dict['roc_auc'] = utility.plot_roc_curve(y_true, y_pred_prob, dataset_name)
        metrics_dict['prec_rec_auc'] = utility.plot_precision_recall_curve(y_true, y_pred_prob, dataset_name)
    else:
        utility.plot_multiclass_roc_curves(y_true_one_hot, y_pred_prob, num_classes, dataset_name)

    metrics = utility.compute_metrics(conf_mat_df.values)
    metrics_dict.update(metrics)

    # logging.info(metrics_dict)
    return metrics_dict


def extract_samples(X, y, class_idx, num_samples):
    X_extracted_all = X[y == class_idx]
    y_extracted_all = y[y == class_idx]

    total_extracted = X_extracted_all.shape[0]
    # print('total_extracted = {}, num_samples = {}'.format(total_extracted, num_samples))
    assert total_extracted >= num_samples   # Ensure dataset has enough samples of the class

    random_idx = np.random.choice(total_extracted, num_samples, replace=False)

    X_extracted = X_extracted_all[random_idx]
    y_extracted = y_extracted_all[random_idx]

    return X_extracted, y_extracted


def prepare_test_train(X_all, y_all, test_size, dataset_percentage, ud_ratio):
    damaged_count = np.count_nonzero(y_all == 1)
    total_need_to_extract = math.floor(X_all.shape[0] * dataset_percentage)
    total_possible_to_extract = math.floor(damaged_count * sum(ud_ratio) / ud_ratio[1])

    total_taken_samples = min(total_need_to_extract, total_possible_to_extract)

    logging.info('Truncating dataset to {:.2f}% of original size, and extracting undamaged and damaged samples'.
                 format(total_taken_samples * 100 / X_all.shape[0]))

    undamaged_samples = int(total_taken_samples * ud_ratio[0] / sum(ud_ratio))
    damaged_samples = total_taken_samples - undamaged_samples

    X_undamaged, y_undamaged = extract_samples(X_all, y_all, class_idx=0, num_samples=undamaged_samples)
    X_damaged, y_damaged = extract_samples(X_all, y_all, class_idx=1, num_samples=damaged_samples)

    X_all = np.concatenate([X_undamaged, X_damaged])
    y_all = np.concatenate([y_undamaged, y_damaged])

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, stratify=y_all, shuffle=True, test_size=test_size)

    utility.print_class_distribution(y_train, "Train")

    dataset = (X_train, y_train), (X_test, y_test)
    return dataset


def prepare_tabular_dataset(dataset):
    logging.info('Preparing tabular data')

    (X_train, y_train), (X_test, y_test) = dataset

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    logging.info('Training set (X_train) shape: {}'.format(X_train.shape))
    logging.info('Test set (X_test) shape: {}'.format(X_test.shape))

    dataset = (X_train, y_train), (X_test, y_test)
    return dataset


def get_filename_from_args():
    num_args = len(sys.argv)
    if num_args != 2:
        print('Usage:')
        print('python run_classifier <experiments_csv_filename>')
        sys.exit(1)

    filename = sys.argv[1]
    isfile = os.path.isfile(filename)

    if not isfile:
        logging.info('File {} does not exist'.format(filename))
        sys.exit()

    return filename


def augment_exp_params(params):
    if params['model'] in ['ann', 'lstm']:  # Set input_nodes for these 2 models based on feature type
        input_nodes = 22    # Assume catch22 by default
        if params['feature_type'] == 'raw_window':  # Other case
            input_nodes = params['window_size']
        input_nodes_key = params['model'] + '_' + 'input_nodes'
        params[input_nodes_key] = input_nodes

    if params['model'] == 'lstm':
        params['lstm_time_steps'] = params['window_size']
        params['lstm_input_nodes'] = 1  # 1-D time series


def set_output_nodes_count(y_all, params):
    unique, counts = np.unique(y_all, return_counts=True)
    num_classes = counts.size

    if num_classes == 2:
        output_nodes = 1
    else:
        output_nodes = num_classes

    params['output_nodes'] = output_nodes


def print_evaluation_metrics(train_metrics, test_metrics):
    metrics_df = pd.DataFrame(columns=train_metrics.keys())
    metrics_df = metrics_df.append(train_metrics, ignore_index=True)
    metrics_df = metrics_df.append(test_metrics, ignore_index=True)

    logging.info('Evaluation metrics (train and test sets) printed below\n\n{}\n'.format(metrics_df))


def run_experiment(params):
    # ---------------------------------------------------
    # Data preparation

    assert params['feature_type'] in ['catch22', 'raw_window']
    use_catch22 = False
    if params['feature_type'] == 'catch22':
        use_catch22 = True

    X_all, y_all = utility.load_prepared_dataset(params, use_catch22)

    set_output_nodes_count(y_all, params)

    dataset = prepare_test_train(X_all, y_all, params['test_set_size'], params['dataset_percentage'], params['ud_ratio'])

    if params['model'] != 'lstm':
        dataset = prepare_tabular_dataset(dataset)

        if params['scale_dataset']:
            dataset = scale_dataset(dataset)

    # ----------------------------------
    # Model training

    model, history = create_and_train_model(params, dataset)
    utility.plot_training_history(history)

    # ----------------------------------
    # Evaluation

    (X_train, y_train), (X_test, y_test) = dataset
    train_metrics = evaluate_model(model, X_train, y_train, "Train_set")
    test_metrics = evaluate_model(model, X_test, y_test, "Test_set")

    print_evaluation_metrics(train_metrics, test_metrics)

    # ----------------------------------
    # Final actions

    utility.save_all_figures(params['results_dir'])
    filename = params['results_dir'] + '/model.pickle'
    utility.save_obj_to_disk(model, filename)
    # plt.show()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU

    # Quick logging setup. Proper logging (to file) is setup later for each experiment
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

    filename = get_filename_from_args()
    experiments = utility.get_experiments(filename)

    for exp_param_set in experiments:
        logging.info('--------------- Running experiment {} --------------- \n'.format(exp_param_set['exp_id']))

        augment_exp_params(exp_param_set)
        exp_param_set['results_dir'] = 'output'    # Set base results dir
        utility.initialize(exp_param_set)
        run_experiment(exp_param_set)

        logging.info('--------------- Finished running experiment {} --------------- \n'.format(exp_param_set['exp_id']))

    logging.info('================= Finished running all experiments ================= \n')


if __name__ == "__main__":
    main()
