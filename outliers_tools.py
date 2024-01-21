from typing import Any

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def detect_outliers_in_column(column: pd.Series, threshold: float = 1.5) -> list:
    """
    Detect outliers in a column of a dataframe using IQR (interquartile range) method
    :param column: column of a dataframe (pd.Series)
    :param threshold: threshold for outlier detection
    :return: list of indices of outliers
    """
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outlier_indices = column[(column < lower_bound) | (column > upper_bound)].index.tolist()

    return outlier_indices


def outliers_labeling(df: pd.DataFrame, threshold: float = 1.5, inplace: bool = False,
                      label: np.nan or Any = np.nan) -> pd.DataFrame:
    """
    Label outliers in a dataframe using detect_outliers_in_column function with selected label
    :param df: raw dataframe which will be modified to include=np.number columns
    :param threshold: threshold for outlier detection
    :param inplace: option to modify the original dataframe or return a new one
    :param label: label for outliers with default value of nan
    :return: dataframe with labeled outliers
    """
    df_working = df.copy()
    df_working = df_working[df_working.select_dtypes(include=np.number).columns]

    for col in df_working.columns:

        if set(df_working[col].unique()) == {0, 1}:
            continue

        outlier_indices = detect_outliers_in_column(df_working[col], threshold)
        df_working.loc[outlier_indices, col] = label

    if inplace:
        df[df_working.columns] = df_working
        return df
    else:
        return df_working


def count_nan_in_row(df: pd.DataFrame, threshold: float = 1.5, inplace: bool = False) -> pd.DataFrame:
    """
    Count number of nan values in each row of a dataframe
    :param df: raw dataframe which will be modified to include=np.number columns
    :param threshold: threshold for outlier detection
    :param inplace: option to modify the original dataframe or return a new one
    :return: dataframe with number of nan values in each row
    """
    df_working = df.copy()
    df_working = df_working[df_working.select_dtypes(include=np.number).columns]

    df_working = outliers_labeling(df_working, threshold)
    df_working['outliers_count'] = df_working.isnull().sum(axis=1)

    if inplace:
        df['outliers_count'] = df_working['outliers_count']
        return df
    else:
        return df_working['outliers_count']


def pca_outliers_count(df: pd.DataFrame, threshold: float = 1.5, inplace=False) -> pd.DataFrame:
    """
    Count number of outliers in each row of a dataframe using PCA
    :param df: raw dataframe which will be modified to include=np.number columns and PCA columns
    :param threshold: threshold for outlier detection
    :param inplace: option to modify the original dataframe or return a new one (after PCA)
    :return: dataframe with number of outliers in each row
    """
    df_working = df.copy()
    df_working_numeric = df_working.select_dtypes(include=np.number)

    pca = PCA(n_components=2)
    df_working_pca = pca.fit_transform(df_working_numeric)
    df_working_pca = pd.DataFrame(df_working_pca, columns=['PCA1', 'PCA2'])

    df_working_pca = outliers_labeling(df_working_pca, threshold)
    df_working_pca['outliers_count'] = df_working_pca.isnull().sum(axis=1)

    if inplace:
        return df_working_pca

    return df_working_pca[['outliers_count']]