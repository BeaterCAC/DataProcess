# @Time    : 2025/1/17 14:06
# @Author  : Aocheng Chen
# @FileName: DataFilling.py
# @Describe: 该模块用于缺失值的填充，包括逻辑填充、相关性填充以及常规填充，异常值的处理同样可以调用该模块。

import pandas as pd
import numpy as np
import warnings
import itertools
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings("ignore")


def logic_filling(df, sheet):
    """
    本函数用于填充部分具有逻辑的变量。
    :param df: pandas DataFrame，包含需要处理的数据
    :param sheet: excel，包含所需处理变量的一些信息
    :return: 填充完毕的DataFrame
    """
    data = df.copy()

    # 提取需要填充的变量信息
    tmpsheet = sheet.loc[pd.notna(sheet['原始数据名']), ['原始数据名', '新数据名', '是否逻辑填充', '逻辑填充']]
    fill_list = list(zip(tmpsheet['新数据名'], tmpsheet['是否逻辑填充'], tmpsheet['逻辑填充']))

    for curv, logicornot, logic in fill_list:
        if logicornot == 0:
            continue  # 不需要逻辑填充
        elif logicornot in [1, 2]:
            # 解析逻辑填充规则
            prev, prevlogit, curvfilling = eval(logic)

            if logicornot == 1:
                # 逻辑填充1：直接填充
                for j in prevlogit:
                    data.loc[data[prev] == j, curv] = curvfilling
            elif logicornot == 2:
                # 逻辑填充2：条件填充
                data[curv] = [
                    curvfilling if (pd.isna(row[curv]) and not pd.isna(row[prev]) and row[prev] not in prevlogit)
                    else row[curv]
                    for _, row in data.iterrows()
                ]
        else:
            print(f"警告：未知的逻辑填充类型 {logicornot}，跳过该变量 {curv}")

    return data


def it_choices(tool_sheet):
    """
    读取多行 'itchoice_list' 字段，解析成数值范围的上界列表。
    :param tool_sheet: pandas DataFrame，包含一个名为 'itchoice_list' 的列，内容为字符串，每行表示一组带编号的区间。
    :return: obj_list，列表格式为 [[5000, 10000, 20000, ...], [...], ...]
    """
    raw_choices = tool_sheet['itchoice_list'].dropna().tolist()

    if len(raw_choices) == 0:
        raise ValueError("tool_sheet['itchoice_list'] 为空，无法生成 it_choices")

    obj_list = []

    for line in raw_choices:
        temp_list = [float("-inf")]
        segments = line.split(', ')
        for seg in segments:
            try:
                value_ = seg[seg.index(' ') + 1:]  # 去掉前面的编号
            except ValueError:
                continue

            if '以下' in value_:
                num = value_[:value_.index('以下')]
                upbound = num.translate(str.maketrans({'千': '000', '万': '0000'}))
                temp_list.append(eval(upbound))
            elif '-' in value_:
                num = value_.split('-')[1]
                upbound = num.translate(str.maketrans({'千': '000', '万': '0000'}))
                temp_list.append(eval(upbound))
            elif '以上' in value_:
                # 无需加，float('inf') 会在外部 append
                continue
        temp_list.append(float("inf"))
        obj_list.append(temp_list)

    return obj_list

def to_it_filling(df, sheet, tool_sheet):
    """
    本函数用于将数值变量填充到对应的范围变量里面。
    :param df: pandas DataFrame，包含需要处理的数据
    :param sheet: excel，包含所需处理变量的一些信息
    :return: 填充完毕的DataFrame
    手动对变量进行分类：
        obj_list[0]---5千以下, 2. 5千-1万, 3. 1万-2万, 4. 2万-5万, 5. 5万-10万, 6. 10万-15万, 7. 15万-20万, 8. 20万-30万, 9. 30万-50万, 10. 50万-100万, 11. 100万以上
            'CshAmt','CshAmt_it','ThirdPartyPayAmt','ThirdPartyPayAmt_it'
        obj_list[1]---1万以下, 2. 1万-3万, 3. 3万-5万, 4. 5万-7万, 5. 7万-10万, 6. 10万-30万, 7. 30万-50万, 8. 50万-100万, 9. 100万-500万, 10. 500万-1000万, 11. 1000万以上
            'ShopAmt','ShopAmt_it','CurrDepbanAmt','CurrDepbanAmt_it','StockV','StockV_it','FundAmt','FundAmt_it','BondAmt','BondAmt_it','GoldAmt','GoldAmt_it'
        obj_list[2]---1. 1万以下, 2. 1万-2万, 3. 2万-5万, 4. 5万-10万, 5. 10万-20万, 6. 20万-30万, 7. 30万-50万, 8. 50万-100万, 9. 100万-200万, 10. 200万-500万, 11. 500万以上
            'CarsPc','CarsPc_it','FicPctAmt','FicPctAmt_it','LoanAmt','LoanAmt_it'
        obj_list[3]---1. 5万以下, 2.5万-10万, 3. 10万-20万, 4. 20万-50万, 5. 50万-100万, 6. 100万-200万, 7. 200万-500万, 8. 500万-1000万, 9. 1000万-2000万, 10. 2000万-5000万, 11. 5000万以上
            'BsAsset','BsAsset_it'
    """
    data = df.copy()
    rgchoice_type = it_choices(tool_sheet)

    # 获取sheet
    tmpsheets = sheet
    my_list = []
    # 保存所有需要填充的变量

    tmpsheet = tmpsheets.loc[
        (pd.notna(sheet['原始数据名']) & pd.notna(sheet['范围与连续填充'])), ['原始数据名', '新数据名',
                                                                              '范围与连续填充']]

    curvar = tmpsheet['新数据名'].tolist()
    pre_itval = tmpsheet['范围与连续填充'].tolist()

    tmplist = list(zip(curvar, pre_itval))
    my_list.extend(tmplist)  # [['a', 1], ['b', 2], ['c', 1], ['d', 3], ['e', 1]]

    # 处理my_list
    lst_copy = [[item[0], f'{item[0]}it', item[1]] for item in my_list]
    lst_merged = sorted(lst_copy, key=lambda x: x[2])
    groups = []
    for k, g in itertools.groupby(lst_merged, lambda x: x[2]):
        groups.append(list(g))  # [[['a', 'ait', 1], ['c', 'cit', 1]], [['b', 'bit', 2]]]

    # 去掉数字
    new_list = []
    for sublist in groups:
        new_sublist = []
        for inner_sublist in sublist:
            new_inner_sublist = inner_sublist[:2]
            new_sublist.append(new_inner_sublist)
        new_list.append(new_sublist)

    # 打开最里面的小列表
    final_list = []
    for sublist in new_list:
        new_sublist = []
        for inner_sublist in sublist:
            for item in inner_sublist:
                new_sublist.append(item)
        final_list.append(new_sublist)  # [['a', 'ait', 'c', 'cit', 'e', 'eit'], ['b', 'bit'], ['d', 'dit']]

    for list_ in range(len(final_list)):
        tmpobj = rgchoice_type[list_]
        tmplist = list(zip([final_list[list_][i] for i in range(len(final_list[list_])) if i % 2 == 0],
                           [final_list[list_][i] for i in range(len(final_list[list_])) if i % 2 == 1]))
        # print(tmplist)
        for i in tmplist:
            LogFil_v = i[1]
            # print(LogFil_v)
            pre_in = i[0]

            # print(pre_in)
            def trans(x, mybins=tmpobj):
                for i in range(0, len(mybins)):
                    try:
                        if x > mybins[i] and x <= mybins[i + 1]:
                            return i + 1
                    except:
                        return i

            data[LogFil_v] = data[pre_in].map(trans)
            print(f'{pre_in}的范围变量{LogFil_v}填充完毕')

    return data


def pre_it_filling(df, sheet, tool_sheet):
    """
    本函数用于填充it变量的前置变量（有些it变量有值，但是前置变量没有值，因此倒过来填充，逻辑为填充范围均值）。
    :param df: pandas DataFrame，包含需要处理的数据
    :param sheet: excel，包含所需处理变量的一些信息
    :return: 填充完毕的DataFrame
    """
    data = df.copy()
    rgchoice_type = it_choices(tool_sheet)

    # 获取sheet
    tmpsheet = sheet.loc[
        (pd.notna(sheet['原始数据名']) & pd.notna(sheet['范围与连续填充'])),
        ['原始数据名', '新数据名', '范围与连续填充']
    ]
    curvar = tmpsheet['新数据名'].tolist()
    pre_itval = tmpsheet['范围与连续填充'].tolist()

    # 构建任务列表
    task_list = [[var, f"{var}_it", pre_val] for var, pre_val in zip(curvar, pre_itval)]
    task_list_sorted = sorted(task_list, key=lambda x: x[2])

    # 按 pre_it_filling 分组
    grouped_tasks = {}
    for _, group in itertools.groupby(task_list_sorted, key=lambda x: x[2]):
        group = list(group)
        grouped_tasks[group[0][2]] = [(item[0], item[1]) for item in group]

    # 计算每个范围的均值
    rgchoice_type2 = []  # 用于储存新的bin,形式为[[2500,5000,7500][1000,3000,4000,5000]]
    for rgchoice in rgchoice_type:
        tmp_list = []
        for i in range(len(rgchoice)):
            if rgchoice[i] == float("-inf"):
                tmp_list.append((0 + rgchoice[1]) / 2)
            elif rgchoice[i] == float("inf"):
                tmp_list.append(rgchoice[-2] + (rgchoice[-2] - rgchoice[-3]) / 2)
            elif rgchoice[i + 1] == float("inf"):
                pass
            else:
                tmp_list.append((rgchoice[i] + rgchoice[i + 1]) / 2)
        rgchoice_type2.append(tmp_list)

    # 填充前置变量
    for group_idx, (group_key, group_vars) in enumerate(grouped_tasks.items()):
        cur_bin = rgchoice_type2[group_idx]
        value_map = {i + 1: mean for i, mean in enumerate(cur_bin)}

        for log_var, it_var in group_vars:
            # 检查 it_var 和 log_var 是否在数据中
            if it_var not in data.columns or log_var not in data.columns:
                print(f"警告：{it_var} 或 {log_var} 不在数据中，跳过该变量")
                continue

            # 填充前置变量
            data.loc[data[log_var].isnull(), log_var] = data[it_var].replace(value_map)
            print(f"IT 变量 {it_var} 的范围值填充到前置变量 {log_var} 完成")

    return data


def filling_tool(data, att=[], method='mean'):
    """
    本函数用于有序变量的均值填充以及无序变量的众数填充。
    :param data: pandas DataFrame，包含需要处理的数据
    :param att: list，需要填充的变量列表
    :param method: str，填充方法，默认为均值填充。
        可选值：'mean', 'zero', 'mode', 'medi', 'cust', 'inter', 'dist'。
    :return: 填充完毕的DataFrame
    """
    df = data.copy()

    if method == 'mean':
        for col in att:
            mean_val = df.loc[df[col] != 0, col].mean()
            df[col].fillna(mean_val, inplace=True)
    elif method == 'zero':
        for col in att:
            df[col].fillna(0, inplace=True)
    elif method == 'mode':
        for col in att:
            mode_val = df.loc[df[col] != 0, col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
    elif method == 'medi':
        for col in att:
            medi_val = df.loc[df[col] != 0, col].median()
            df[col].fillna(medi_val, inplace=True)
    elif method == 'cust':
        cust_val = float(input("请输入需要填充的值："))
        for col in att:
            df[col].fillna(cust_val, inplace=True)
    elif method == 'inter':
        for col in att:
            df[col] = df[col].interpolate()
    elif method == 'dist':
        dist_ = float(input("请输入需要填充的分位数值："))
        for col in att:
            sort_ = df.loc[df[col].notnull() & (df[col] != 0), col].sort_values().reset_index(drop=True)
            len_ = len(sort_)
            index_ = int(len_ * dist_)
            dist_val = sort_[index_]
            df[col].fillna(dist_val, inplace=True)
    else:
        print(f"错误：'{method}' 填充方法不存在，退出")

    return df


def norm_filling(df, sheet, group_col=None):
    """
    本函数用于普通填充，支持以下填充方法：
    0: 0值填充
    1: 众数填充
    2: 均值填充
    3: 基于同类样本的中心度量填充
    4: 基于推断的方法填充（回归、决策树等）
    :param data: pandas DataFrame，包含需要处理的数据
    :param sheet: excel，包含填充规则的规则表
    :param method: str，填充方法，默认为均值填充。
        可选值：'mean', 'zero', 'mode', 'medi', 'cust', 'inter', 'dist'。
    :param group_col: str，用于分组的列（例如类别列），默认为 None。
    :return: 填充完毕的DataFrame
    """
    data = df.copy()

    # 获取sheet
    tmpsheet = sheet.loc[pd.notna(sheet['原始数据名']), ['原始数据名', '新数据名', '常规填充']]
    fill_list = list(zip(tmpsheet['新数据名'], tmpsheet['常规填充']))

    for curv, filling_type in fill_list:
        if filling_type == 0:
            # 0值填充
            data = filling_tool(data, att=[curv], method='zero')
        elif filling_type == 1:
            # 众数填充
            data = filling_tool(data, att=[curv], method='mode')
        elif filling_type == 2:
            # 均值填充
            data = filling_tool(data, att=[curv], method='mean')
        elif filling_type == 3:
            # 基于同类样本的中心度量填充
            if group_col is None:
                print(f"警告：未提供分组列，无法对变量 {curv} 进行基于同类样本的填充")
            else:
                data = fill_by_group_mean(data, curv, group_col)
        elif filling_type == 4:
            feature_str = sheet.loc[sheet['新数据名'] == curv, '推断用特征'].values
            method_str = sheet.loc[sheet['新数据名'] == curv, '推断方法'].values

            if len(feature_str) == 0 or pd.isna(feature_str[0]):
                print(f"⚠️ 警告：未指定用于变量 {curv} 推断的特征列，跳过该变量")
                continue

            feature_cols = [f.strip() for f in feature_str[0].split(',')]

            method = 'LinearRegression'  # 默认方法
            if len(method_str) > 0 and pd.notna(method_str[0]):
                method = method_str[0].strip()

            try:
                data = fill_by_inference(data, target_col=curv, feature_cols=feature_cols, method=method)
            except Exception as e:
                print(f"❌ 错误：填充变量 {curv} 时失败，原因：{e}")
        else:
            print(f"⚠️ 警告：未知的填充类型 {filling_type}，跳过变量 {curv}")

    return data


def fill_by_group_mean(data, col, group_col):
    """
    基于同类样本的中心度量填充。
    :param data: pandas DataFrame，包含需要处理的数据
    :param col: str，需要填充的目标列
    :param group_col: str，用于分组的列,即为同一类别的类。
    :return: 填充完毕的DataFrame
    """
    if group_col not in data.columns:
        print(f"警告：分组列 {group_col} 不存在，无法进行基于同类样本的填充")
        return data

    # 计算每个组的均值
    group_means = data.groupby(group_col)[col].transform('mean')

    # 填充缺失值
    data[col].fillna(group_means, inplace=True)
    print(f"变量 {col} 的缺失值已使用同类样本的均值填充（分组列：{group_col}）")

    return data


def fill_by_inference(data, target_col, feature_cols, method='LinearRegression'):
    """
    使用模型推断填充指定列中的缺失值。

    :param data: DataFrame，包含数据
    :param target_col: str，要填充缺失值的列名
    :param feature_cols: list[str]，用于推断的其他列名
    :param method: str，模型名称，如 'LinearRegression', 'LGBMRegressor' 等
    :return: 填充完毕的 DataFrame
    """
    model_map = {
        'LinearRegression': LinearRegression,
        'LogisticRegression': LogisticRegression,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'RandomForestClassifier': RandomForestClassifier,
        'LGBMRegressor': LGBMRegressor,
        'LGBMClassifier': LGBMClassifier,
        'XGBRegressor': XGBRegressor,
        'XGBClassifier': XGBClassifier,
    }

    if method not in model_map:
        raise ValueError(f"method 参数必须是以下之一: {list(model_map.keys())}")

    if not feature_cols:
        raise ValueError("feature_cols 参数不能为空")

    # 判断目标变量是否为分类变量
    is_classification = pd.api.types.is_categorical_dtype(data[target_col]) or data[target_col].dtype == 'object'

    known_data = data[data[target_col].notnull()].copy()
    unknown_data = data[data[target_col].isnull()].copy()

    if known_data.empty:
        print(f"⚠️ 变量 {target_col} 没有已知值，无法进行推断填充")
        return data

    # 编码分类变量
    le = None
    if is_classification:
        le = LabelEncoder()
        known_data[target_col] = le.fit_transform(known_data[target_col].astype(str))

    # 初始化并拟合模型
    model = model_map[method]()
    model.fit(known_data[feature_cols], known_data[target_col])

    # 预测缺失值
    predicted = model.predict(unknown_data[feature_cols])

    if is_classification and le is not None:
        predicted = le.inverse_transform(predicted.astype(int))

    data.loc[data[target_col].isnull(), target_col] = predicted

    print(f"✅ 已使用模型 {method} 填充变量 {target_col} 的缺失值")
    return data


def check_miss(df, miss_rate=False):
    """
    本函数用于查看变量缺失值数量、缺失比例的基本信息。
    参数：
    :param df: pandas DataFrame，包含需要处理的数据
    :param miss_rate: list，是否挑选出超过（不足）某比例的缺失值变量名。
        例如：[0.1, 0.9] 表示输出缺失比例小于0.1和大于0.9的变量。
    """
    data = df.copy()
    val_list = data.columns.tolist()
    data_ = data[val_list]

    if miss_rate and (len(miss_rate) != 2 or miss_rate[0] < 0 or miss_rate[1] > 1):
        print("变量的缺省比例范围输入错误，退出函数！")
        return

    # 统计缺失值数量和比例
    missing = data_.isnull().sum().reset_index().rename(columns={0: 'missNum'})
    missing['missRate'] = missing['missNum'] / data_.shape[0]
    miss_analy = missing[missing.missRate > 0].sort_values(by='missRate', ascending=False)

    if miss_analy.shape[0] == 0:
        print("无缺失变量")
        return None

    print(miss_analy)

    if miss_rate:
        highv = miss_analy.loc[miss_analy['missRate'] >= miss_rate[1], 'index'].tolist()
        lowv = miss_analy.loc[miss_analy['missRate'] <= miss_rate[0], 'index'].tolist()
        if miss_rate[0] == 0:
            print(f"\n缺失值比例高于{miss_rate[1]}的变量名有：\n", highv)
        elif miss_rate[1] == 1:
            print(f"\n缺失值比例低于{miss_rate[0]}的变量名有：\n", lowv)
        else:
            print(f"\n缺失值比例低于{miss_rate[0]}的变量名有：\n", lowv)
            print(f"\n缺失值比例高于{miss_rate[1]}的变量名有：\n", highv)

    return miss_analy