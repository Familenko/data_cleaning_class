import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm


def early_stopping(
        X_train,
        y_train,
        X_val,
        y_val,
        estimator=GradientBoostingRegressor(),
        parameter="n_estimators",
        min_n=2,
        max_n=120,
        trial=5,
        metric=mean_squared_error
):
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


def tuning(X_train=None, y_train=None, X_valid=None, y_valid=None,
           estimator='KNeighborsClassifier', target='n_neighbors',
           min_n=1, max_n=30, step=1,
           set_target=None, average='macro', **params):

    params = estimator.get_params()
    if params and target in params:
        del params[target]

    if set_target:
        params_dict = {target: set_target}
        if params:
            params_dict.update(params)
        estimator.set_params(**params_dict)
        estimator.fit(X_train, y_train)
        return estimator

    else:
        metrics = ['precision', 'recall', 'f1', 'accuracy']
        test_error_rates = {metric: [] for metric in metrics}

        for i in tqdm(np.arange(min_n, max_n, step), desc=f"Checking model for {target}"):
            params_dict = {target: i}
            if params:
                params_dict.update(params)
            estimator.set_params(**params_dict)

            estimator.fit(X_train, y_train)
            y_pred_valid = estimator.predict(X_valid)

            test_error_rates['accuracy'].append(accuracy_score(y_valid, y_pred_valid))
            test_error_rates['precision'].append(precision_score(y_valid, y_pred_valid, average=average))
            test_error_rates['recall'].append(recall_score(y_valid, y_pred_valid, average=average))
            test_error_rates['f1'].append(f1_score(y_valid, y_pred_valid, average=average))

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()

        axs[0].plot(np.arange(min_n, max_n, step), test_error_rates['precision'])
        axs[0].set_ylabel('precision')
        axs[0].set_xlabel(f'{target}')
        axs[0].grid(True)

        axs[1].plot(np.arange(min_n, max_n, step), test_error_rates['recall'])
        axs[1].set_ylabel('recall')
        axs[1].set_xlabel(f'{target}')
        axs[1].grid(True)

        axs[2].plot(np.arange(min_n, max_n, step), test_error_rates['f1'])
        axs[2].set_ylabel('f1')
        axs[2].set_xlabel(f'{target}')
        axs[2].grid(True)

        axs[3].plot(np.arange(min_n, max_n, step), test_error_rates['accuracy'])
        axs[3].set_ylabel('accuracy')
        axs[3].set_xlabel(f'{target}')
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()
