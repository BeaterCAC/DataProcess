# @Time    : 2025/1/26 13:52
# @Author  : Aocheng Chen
# @FileName: DataTransformation.py
# @Describe: 该文件用于数据变换，包含规范化/标准化处理以及离散化

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def find_bin(value, bins):
    """
    查找数值对应的分箱索引 (左闭右开区间)
    :param value: pandas DataFrame，需要分箱的数值
    :param bins: list，分箱区间列表，格式如 [(low1, high1), (low2, high2), ...]
    :return: 分箱索引 (从0开始)，未找到返回-1
    """
    for idx, (low, high) in enumerate(bins):
        if low <= value < high:
            return idx
    return -1


def bin_numeric_data(data,feature,method,bin_config):
    """
    数值数据分箱方法
    :param data: pandas DataFrame，包含待分箱特征的DataFrame
    :param feature: 需要分箱的特征列名
    :param method: 分箱方法，可选:
            - 'equal_width' : 等距分箱
            - 'equal_freq'  : 等频分箱
            - 'custom'      : 自定义边界分箱
            - 'quantile'    : 按分位数分箱
    :param bin_config: 分箱配置
            - 等距/等频: 分箱数量 (int)
            - 自定义/分位数: 边界列表 (List[float])
    :return: 包含新增分箱列的DataFrame
    """
    df = data.copy()

    if method == 'equal_width':
        # 等距分箱 (处理浮点数边界)
        min_val = df[feature].min()
        max_val = df[feature].max()
        n_bins = bin_config if isinstance(bin_config, int) else 4
        edges = np.linspace(min_val, max_val, num=n_bins + 1)
        bins = list(zip(edges[:-1], edges[1:]))
        df[f'{feature}_bin'] = df[feature].map(lambda x: find_bin(x, bins))

    elif method == 'equal_freq':
        # 等频分箱 (自动处理重复值)
        n_bins = bin_config if isinstance(bin_config, int) else 4
        df[f'{feature}_bin'] = pd.qcut(
            df[feature],
            q=n_bins,
            duplicates='drop',
            labels=False
        )

    elif method == 'custom':
        # 自定义边界分箱
        if not isinstance(bin_config, list):
            raise ValueError("自定义边界分箱需要传入列表")
        df[f'{feature}_bin'] = pd.cut(
            df[feature],
            bins=bin_config,
            include_lowest=True,
            labels=False
        )
        df[f'{feature}_bin'] = df[f'{feature}_bin'] + 1

    elif method == 'quantile':
        # 分位数分箱
        quantiles = np.quantile(df[feature], bin_config)
        df[f'{feature}_bin'] = pd.cut(
            df[feature],
            bins=np.unique(quantiles),  # 去重处理
            include_lowest=True,
            labels=False
        )

    else:
        raise ValueError(f"不支持的分箱方法: {method}")

    return df


def ordered_bin(df, sheet):
    """
    支持灵活配置的有序变量分箱函数
    :param df: pandas DataFrame，包含待分箱特征的DataFrame
    :param sheet: 配置表，包含列:
        - '新数据名': 列名
        - '有序变量分箱': 分箱规则，可为 0、分箱方法字符串、或边界列表等
    :return: 添加分箱列后的 DataFrame
    """
    data = df.copy()
    tmpsheet = sheet.loc[pd.notna(sheet['有序变量分箱']), ['新数据名', '有序变量分箱']]

    for feature, rule in zip(tmpsheet['新数据名'], tmpsheet['有序变量分箱']):
        # 跳过不需要分箱的变量（例如 0 或 '0'）
        if rule in [0, '0']:
            continue

        # 初始化
        method = None
        bin_config = None

        # 尝试解析字符串
        try:
            # rule 是字符串
            if isinstance(rule, str):
                if ':' in rule:
                    method_part, config_part = rule.split(':', 1)
                    method = method_part.strip()

                    if method == 'custom':
                        bin_config = eval(config_part)
                    elif method == 'quantile':
                        bin_config = eval(config_part)
                    elif method in ['equal_width', 'equal_freq']:
                        bin_config = int(config_part)
                    else:
                        print(f"不支持的分箱方法 {method}，特征 {feature} 跳过。")
                        continue
                elif rule in ['equal_width', 'equal_freq', 'quantile']:
                    method = rule
                    bin_config = 4  # 默认箱数
                else:
                    # 默认尝试作为 custom 列表
                    method = 'custom'
                    bin_config = eval(rule)

            # rule 是列表或数字
            elif isinstance(rule, list):
                method = 'custom'
                bin_config = rule
            elif isinstance(rule, int):
                method = 'equal_width'
                bin_config = rule
            else:
                print(f"无法识别的分箱规则 {rule}，特征 {feature} 跳过。")
                continue

            # 调用分箱
            data = bin_numeric_data(data, feature, method=method, bin_config=bin_config)

        except Exception as e:
            print(f"特征 {feature} 分箱失败：{e}")
            continue

    return data



def unordered_bin(data, sheet):
    """
    本函数用于乱序的合并变量值，从sheet中获取合并规则
    :param data: pandas DataFrame，原始数据
    :param sheet: 合并配置表，需包含列:
        - '新数据名' : 新数据名
        - '乱序合并' : 合并规则字典 (如{1:1,2:1,3:2})
    :return: 处理后的DataFrame
    """
    df = data.copy()
    tmpsheet = sheet.loc[(pd.notna(sheet['无序变量分箱'])), ['新数据名', '无序变量分箱']]

    merge_lists = []
    curvar = tmpsheet['新数据名'].tolist()
    merge_rules = tmpsheet['无序变量分箱'].tolist()
    merge_lists = list(zip(curvar, merge_rules))

    for merge_list in merge_lists:
        att = merge_list[0]
        value_dict = eval(merge_list[1])  # 使用eval将字符串转换为字典
        df[att] = df[att].replace(value_dict)

    return df

# -------------------
# 规范化方法字典
# -------------------
NORMALIZATION_METHODS = {
    1: "minmax",
    2: "zscore",
    3: "decimal_scaling",
    4: "log_transform",
    5: "robust_scaling",
    6: "max_abs_scaling"
}


def apply_normalization(series, method):
    """
    核心规范化函数
    :param series: 待处理的特征列
    :param method: 规范化方法编号 (1-6)
    :return: 处理后的Series
    """
    method_name = NORMALIZATION_METHODS.get(method)
    if not method_name:
        raise ValueError(f"未知的规范化方法编号: {method}")

    if method == 1:  # 最小最大规范化
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    elif method == 2:  # Z分数标准化
        return (series - series.mean()) / series.std()

    elif method == 3:  # 小数定标法
        max_abs = series.abs().max()
        scale = 10 ** np.ceil(np.log10(max_abs)) if max_abs != 0 else 1
        return series / scale

    elif method == 4:  # 对数变换
        if (series <= 0).any():
            raise ValueError("对数变换需要数据均为正值")
        return np.log1p(series)

    elif method == 5:  # 稳健标准化
        iqr = series.quantile(0.75) - series.quantile(0.25)
        return (series - series.median()) / iqr if iqr != 0 else series

    elif method == 6:  # 最大绝对值缩放
        max_abs = series.abs().max()
        return series / max_abs if max_abs != 0 else series

def normalize_data(df, sheet, output_suffix):
    """
    自动化数据规范化处理
    :param df: 原始数据DataFrame
    :param sheet: 配置表，需包含列:
            - '新数据名': 新数据名（作为原始列名）
            - '是否规范化': 0/1 是否执行规范化
            - '规范化形式': 方法编号 (1-6)
    :param output_suffix: 输出列后缀
    :return: 处理后的DataFrame
    """
    processed_df = df.copy()

    # 提取有效配置
    config = sheet.loc[
        (sheet['是否规范化'] == 1) &
        pd.notna(sheet['规范化形式']),
        ['新数据名', '规范化形式']
    ]

    for _, row in config.iterrows():
        orig_col = row['新数据名']
        new_col = f"{orig_col}{output_suffix}"
        method = int(row['规范化形式'])

        try:
            if orig_col not in processed_df.columns:
                print(f"警告: 列 {orig_col} 不存在，已跳过")
                continue

            processed_df[new_col] = apply_normalization(
                processed_df[orig_col],
                method
            )
            print(f"成功处理列: {orig_col} -> {new_col} ({NORMALIZATION_METHODS[method]})")

        except Exception as e:
            print(f"处理列 {orig_col} 时发生错误: {str(e)}")
            continue

    return processed_df

