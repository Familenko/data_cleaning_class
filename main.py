import pandas as pd
import numpy as np


def detect_outliers_in_column(column, threshold=1.5):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outlier_indices = column[(column < lower_bound) | (column > upper_bound)].index.tolist()

    return outlier_indices


def outliers_labeling(df, threshold=1.5, inplace=False, label=np.nan):
    df_working = df.copy()
    df_working = df_working[df_working.select_dtypes(include=np.number).columns]

    for col in df_working.columns:

        if set(df_working[col].unique()) == {0, 1}:
            continue

        outlier_indices = detect_outliers_in_column(df_working[col], threshold)
        df_working.loc[outlier_indices, col] = label

    if inplace:
        df[df_working.columns] = df_working

    return df_working


def count_outliers_in_row(df, threshold=1.5, inplace=False):
    df_working = df.copy()
    df_working = df_working[df_working.select_dtypes(include=np.number).columns]

    df_working = outliers_labeling(df_working, threshold)
    df_working['outliers_count'] = df_working.isnull().sum(axis=1)

    if inplace:
        df['outliers_count'] = df_working['outliers_count']

    return df_working['outliers_count']