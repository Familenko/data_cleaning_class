from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


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

    for n in range(min_n, max_n):
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
