from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tqdm import tqdm


def early_stopping(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        estimator: Callable = GradientBoostingRegressor(),
        parameter: str = "n_estimators",
        min_n: int = 2,
        max_n: int = 120,
        trial: int = 5,
        metric: Callable = mean_squared_error
):
    """
    Early stopping for sklearn estimators during training
    :param X_train: training data
    :param y_train: training labels
    :param X_val: validation data
    :param y_val: validation labels
    :param estimator: sklearn estimator
    :param parameter: parameter to be tuned
    :param min_n: minimum value of parameter
    :param max_n: maximum value of parameter
    :param trial: number of trials to wait before stopping
    :param metric: metric to be used for early stopping
    :return: estimator with optimal parameter
    """
    min_val_error = float("inf")
    error_going_up = 0

    for n in tqdm(range(min_n, max_n), desc=f"Checking model for {parameter}"):
        setattr(estimator, parameter, n)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        val_error = metric(y_val, y_pred)

        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == trial:
                break

    updated_parameter = n - error_going_up
    setattr(estimator, parameter, updated_parameter)
    estimator.fit(X_train, y_train)

    print(getattr(estimator, parameter), "is the optimal value for", parameter)
    print("Minimum validation metric:", min_val_error)

    return estimator


def tuning_classifier(
        X_train: np.array,
        y_train: np.array,
        X_valid: np.array,
        y_valid: np.array,
        estimator: Callable = KNeighborsClassifier(),
        parameter: str = 'n_neighbors',
        min_n: int = 2,
        max_n: int = 120,
        step: int = 1,
        set_parameter: int or None = None
):
    """
    Tuning hyperparameters for sklearn estimators
    :param X_train: training data
    :param y_train: training labels
    :param X_valid: validation data
    :param y_valid: validation labels
    :param estimator: estimator to be tuned
    :param parameter: parameter to be tuned
    :param min_n: minimum value of parameter
    :param max_n: maximum value of parameter
    :param step: step size for parameter
    :param set_parameter: final value of parameter for building final model
    :return: estimator with optimal parameter
    """
    if set_parameter:
        setattr(estimator, parameter, set_parameter)
        estimator.fit(X_train, y_train)
        return estimator

    else:
        metrics = ['precision', 'recall', 'f1', 'accuracy']
        test_error_rates = {metric: [] for metric in metrics}
        total_test_error_rates = {}

        for i in tqdm(np.arange(min_n, max_n, step), desc=f"Checking model for {parameter}"):
            setattr(estimator, parameter, i)
            estimator.fit(X_train, y_train)
            y_pred_valid = estimator.predict(X_valid)

            accuracy = accuracy_score(y_valid, y_pred_valid)
            precision = precision_score(y_valid, y_pred_valid)
            recall = recall_score(y_valid, y_pred_valid)
            f1 = f1_score(y_valid, y_pred_valid)
            total = accuracy + precision + recall + f1

            test_error_rates['accuracy'].append(accuracy)
            test_error_rates['precision'].append(precision)
            test_error_rates['recall'].append(recall)
            test_error_rates['f1'].append(f1)
            total_test_error_rates[i] = total

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()

        best_metric_summary = {metric: max(test_error_rates[metric]) for metric in metrics}
        best_metric = max(best_metric_summary, key=best_metric_summary.get)
        best_metric_value = best_metric_summary[best_metric]
        best_parameter = np.arange(min_n, max_n, step)[test_error_rates[best_metric].index(best_metric_value)]
        best_optimal_parameter = max(total_test_error_rates, key=total_test_error_rates.get)

        print(f"Best {best_metric} value: {best_metric_value} with {parameter} value: {best_parameter}")
        print(f"Best optimal {parameter} value: {best_optimal_parameter}")

        axs[0].plot(np.arange(min_n, max_n, step), test_error_rates['precision'])
        axs[0].axvline(x=best_optimal_parameter, color='red', linestyle='--', label=f'Best {parameter}')
        axs[0].set_ylabel('precision')
        axs[0].set_xlabel(f'{parameter}')
        axs[0].grid(True)

        axs[1].plot(np.arange(min_n, max_n, step), test_error_rates['recall'])
        axs[1].axvline(x=best_optimal_parameter, color='red', linestyle='--', label=f'Best {parameter}')
        axs[1].set_ylabel('recall')
        axs[1].set_xlabel(f'{parameter}')
        axs[1].grid(True)

        axs[2].plot(np.arange(min_n, max_n, step), test_error_rates['f1'])
        axs[2].axvline(x=best_optimal_parameter, color='red', linestyle='--', label=f'Best {parameter}')
        axs[2].set_ylabel('f1')
        axs[2].set_xlabel(f'{parameter}')
        axs[2].grid(True)

        axs[3].plot(np.arange(min_n, max_n, step), test_error_rates['accuracy'])
        axs[3].axvline(x=best_optimal_parameter, color='red', linestyle='--', label=f'Best {parameter}')
        axs[3].set_ylabel('accuracy')
        axs[3].set_xlabel(f'{parameter}')
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()


def tuning_regressor(
        X_train: np.array,
        y_train: np.array,
        X_valid: np.array,
        y_valid: np.array,
        estimator: Callable = KNeighborsRegressor(),
        parameter: str = 'n_neighbors',
        min_n: int = 2,
        max_n: int = 120,
        step: int = 1,
        set_parameter: int or None = None
):
    """
    Tuning hyperparameters for sklearn regression estimators
    :param X_train: training data
    :param y_train: training labels
    :param X_valid: validation data
    :param y_valid: validation labels
    :param estimator: estimator to be tuned
    :param parameter: parameter to be tuned
    :param min_n: minimum value of parameter
    :param max_n: maximum value of parameter
    :param step: step size for parameter
    :param set_parameter: final value of parameter for building the final model
    :return: estimator with optimal parameter
    """
    if set_parameter:
        setattr(estimator, parameter, set_parameter)
        estimator.fit(X_train, y_train)
        return estimator

    else:
        metrics = ['mean_absolute_error', 'mean_squared_error', 'r2_score', 'median_absolute_error']
        test_error_rates = {metric: [] for metric in metrics}
        total_test_error_rates = {}

        for i in tqdm(np.arange(min_n, max_n, step), desc=f"Checking model for {parameter}"):
            setattr(estimator, parameter, i)
            estimator.fit(X_train, y_train)
            y_pred_valid = estimator.predict(X_valid)

            mae = mean_absolute_error(y_valid, y_pred_valid)
            mse = mean_squared_error(y_valid, y_pred_valid)
            r2 = r2_score(y_valid, y_pred_valid)
            medae = median_absolute_error(y_valid, y_pred_valid)

            test_error_rates['mean_absolute_error'].append(mae)
            test_error_rates['mean_squared_error'].append(mse)
            test_error_rates['r2_score'].append(r2)
            test_error_rates['median_absolute_error'].append(medae)
            total_test_error_rates[i] = r2

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()

        best_optimal_parameter = max(total_test_error_rates, key=total_test_error_rates.get)

        print(f"Best optimal {parameter} value: {best_optimal_parameter}")

        axs[0].plot(np.arange(min_n, max_n, step), test_error_rates['mean_absolute_error'])
        axs[0].axvline(x=best_optimal_parameter, color='red', linestyle='--', label=f'Best {parameter}')
        axs[0].set_ylabel('Mean Absolute Error')
        axs[0].set_xlabel(f'{parameter}')
        axs[0].grid(True)

        axs[1].plot(np.arange(min_n, max_n, step), test_error_rates['mean_squared_error'])
        axs[1].axvline(x=best_optimal_parameter, color='red', linestyle='--', label=f'Best {parameter}')
        axs[1].set_ylabel('Mean Squared Error')
        axs[1].set_xlabel(f'{parameter}')
        axs[1].grid(True)

        axs[2].plot(np.arange(min_n, max_n, step), test_error_rates['r2_score'])
        axs[2].axvline(x=best_optimal_parameter, color='red', linestyle='--', label=f'Best {parameter}')
        axs[2].set_ylabel('R2 Score')
        axs[2].set_xlabel(f'{parameter}')
        axs[2].grid(True)

        axs[3].plot(np.arange(min_n, max_n, step), test_error_rates['median_absolute_error'])
        axs[3].axvline(x=best_optimal_parameter, color='red', linestyle='--', label=f'Best {parameter}')
        axs[3].set_ylabel('Median Absolute Error')
        axs[3].set_xlabel(f'{parameter}')
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()
