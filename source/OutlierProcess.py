# @Time    : 2025/1/20 21:44
# @Author  : Aocheng Chen
# @FileName: OutlierProcess.py
# @Describe: 本文件针对数据中的异常值进行处理，包含有判断异常值、删除异常值、修正异常值等功能

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from DataProcess_2025.DataFilling import filling_tool, fill_by_group_mean, fill_by_inference


def detect_outliers(df, method='boxplot'):
    """
    判断异常值的函数
    :param df: pandas DataFrame，包含需要处理的数据
    :param method: str，选择判断异常值的方法，可选 '箱型图', 'z分数', '散点图', '聚类'
    :return: 布尔型的DataFrame，表示每个数据点是否为异常值
    """
    if method == 'boxplot':
        # 箱型图法
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

    elif method == 'zscore':
        # Z分数法
        z_scores = np.abs(stats.zscore(df))
        return z_scores > 3

    elif method == 'scatter':
        # 散点图法（假设只处理两列数据）
        if df.shape[1] != 2:
            raise ValueError("散点图法只适用于两列数据")
        # 计算点到中心的距离，超过某个阈值则为异常值
        center = df.mean()
        distances = np.sqrt(((df - center) ** 2).sum(axis=1))
        return distances > distances.quantile(0.95)

    elif method == 'clustering':
        # 聚类法（使用DBSCAN）
        clustering = DBSCAN(eps=3, min_samples=2).fit(df)
        return clustering.labels_ == -1

    else:
        raise ValueError("未知的异常值检测方法")


def remove_outliers(df, method='箱型图'):
    """
    删除异常值的函数
    :param df: pandas DataFrame，包含需要处理的数据
    :param method: str，选择判断异常值的方法，可选 '箱型图', 'z分数', '散点图', '聚类'
    :return: 删除异常值后的DataFrame
    """
    outliers = detect_outliers(df, method)
    return df[~outliers.any(axis=1)]


def correct_outliers(df, method='boxplot', fill_method='mean', group_col=None):
    """
    修正异常值的方法，使用填充方法替换异常值。
    参数：
    :param df: pd.DataFrame需要处理的数据框。
    :param method: str，检测异常值的方法，默认为 'boxplot'。
        可选值：'boxplot', 'zscore', 'scatter', 'clustering'。
    :param fill_method: str，填充方法，默认为 'mean'。
        可选值：'mean', 'zero', 'mode', 'medi', 'cust', 'inter', 'dist'。
    :param sheet: pd.DataFrame，包含填充规则的规则表，默认为 None。
    :param group_col: str，用于分组的列（例如类别列），默认为 None。
    :return: pd.DataFrame，修正异常值后的数据框。
    """
    # 检测异常值
    outliers = detect_outliers(df, method)

    # 复制数据框以避免修改原始数据
    data = df.copy()

    # 遍历每一列，用填充方法替换异常值
    for col in data.columns:
        if col in outliers.columns:
            # 获取异常值的索引
            outlier_indices = outliers[col]

            # 如果使用普通填充方法
            if fill_method in ['mean', 'zero', 'mode', 'medi', 'cust', 'inter', 'dist']:
                # 调用现有的 filling_tool 方法进行填充
                filled_data = filling_tool(data, att=[col], method=fill_method)
                # 只替换异常值
                data.loc[outlier_indices, col] = filled_data.loc[outlier_indices, col]

            # 如果使用基于同类样本的填充
            elif fill_method == 'group_mean' and group_col is not None:
                # 调用 fill_by_group_mean 方法进行填充
                filled_data = fill_by_group_mean(data, col, group_col)
                # 只替换异常值
                data.loc[outlier_indices, col] = filled_data.loc[outlier_indices, col]

            # 如果使用基于推断的填充
            elif fill_method in ['regression', 'decision_tree']:
                # 调用 fill_by_inference 方法进行填充
                filled_data = fill_by_inference(data, col, method=fill_method)
                # 只替换异常值
                data.loc[outlier_indices, col] = filled_data.loc[outlier_indices, col]

            else:
                print(f"警告：未知的填充方法 {fill_method}，跳过该列 {col}")

    return data