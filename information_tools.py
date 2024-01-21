import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def cv_results(estimator, result='df'):
    """
    Return dataframe with cross validation results
    :param estimator: CV estimator
    :param result: option to return dataframe or plot
    :return: dataframe with cross validation results
    """
    results = pd.DataFrame(estimator.cv_results_)
    parameter_names = list(results['params'][0].keys())
    parameter_names = ['param_' + param for param in parameter_names]
    parameter_names.append('mean_test_score')
    parameter_names.append('std_test_score')
    parameter_names.append('params')
    results.sort_values(by='mean_test_score', ascending=False, inplace=True)
    results.reset_index(drop=True, inplace=True)

    if result == 'df':
        return results[parameter_names]

    if result == 'plot':
        results['mean_test_score'].plot(
            yerr=[results['std_test_score'],
                  results['std_test_score']],
            subplots=True)
        plt.ylabel('Mean test score')
        plt.xlabel('Hyperparameter combinations')
        plt.grid(True)
        plt.show();


def plot_pca(df, variance=0.95, svd_solver='auto', inplace=None):
    """
    Plot PCA explained variance ratio in relation to number of dimensions
    :param df: raw dataframe which will be modified
    :param variance: variance threshold
    :param svd_solver: svd solver for PCA
    :param inplace: option to modify the original dataframe with selected number of dimensions
    :return: dataframe with selected number of dimensions
    """

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA(svd_solver=svd_solver)
    pca.fit(df_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    dimension = np.argmax(cumsum >= variance) + 1

    print(f'Variance - {variance}, Dimension - {dimension}')

    explained_variance = []
    dimension = []

    for v in np.arange(0.01, 1.0, 0.05):
        d = np.argmax(cumsum >= v) + 1

        explained_variance.append(v)
        dimension.append(d)

    if inplace:
        pca = PCA(n_components=inplace)
        df_pca = pca.fit_transform(df_scaled)
        df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
        return df_pca

    plt.plot(dimension, explained_variance)
    plt.xlabel("Number of Dimension")
    plt.ylabel("Variance Explained")
    plt.grid(alpha=0.2)
    plt.show();
