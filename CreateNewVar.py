# @Time    : 2025/3/1 14:27
# @Author  : Aocheng Chen
# @FileName: CreateNewVar.py
# @Describe: 本文件用于构造新变量

#导入基础库
import pandas as pd
import numpy as np
import warnings
import copy

#from regex import E
warnings.filterwarnings("ignore")


def hold_var(data, sheet):
    df = data.copy()

    tmpsheets =sheet

    tmpsheet = tmpsheets.loc[(pd.notna(tmpsheets['持有'])),['新数据名','持有']]
    curvar = tmpsheet['新数据名'].tolist()

    for holdx in curvar:
        df[holdx[:-1]] = df[holdx].apply(lambda x :1 if x>0 else 0)

    return df

def apply_rule_conditions(ind, new_var_sheet):
    """
    根据规则表 new_var_sheet，对 ind 数据按家庭（hhid）聚合统计，生成新变量列（含占比）。

    参数：
    - ind: DataFrame，个体级数据，包含 'hhid' 和相关变量列
    - new_var_sheet: DataFrame，需包含列 ['原始数据名', '规则', '新生成变量名']

    返回：
    - merged_df: DataFrame，包含每个家庭的统计变量和对应占比（分母为家庭规模）
    """

    # 替换缺失值为 NaN
    ind.replace(['.d', '.r', '.e', '.n'], np.nan, inplace=True)

    # 类型转换
    for col in new_var_sheet['原始数据名'].unique():
        if col in ind.columns:
            ind[col] = ind[col].apply(lambda x: float(x) if not pd.isnull(x) else x)

    result_list = []

    # 家庭规模
    family_size = ind.groupby('hhid')['pline'].count().reset_index().rename(columns={'pline': '家庭规模'})
    result_list.append(family_size)

    # 存储计数字段的列名
    count_var_names = []

    # 遍历规则表生成统计列
    for _, row in new_var_sheet.iterrows():
        col = row['原始数据名']
        rule_str = row['规则']
        new_col = row['新生成变量名']
        count_var_names.append(new_col)

        # 多条件解析
        conditions = [cond.strip() for cond in rule_str.split(',')]
        query = ' & '.join([f"(ind['{col}'] {cond})" for cond in conditions])
        query += f" & pd.notna(ind['{col}'])"

        try:
            filtered = ind.loc[eval(query)]
            grouped = filtered.groupby('hhid')[col].count().reset_index().rename(columns={col: new_col})
            result_list.append(grouped)
        except Exception as e:
            print(f"解析规则失败: {rule_str} -> {e}")

    # 合并所有变量（左连接）
    from functools import reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='hhid', how='outer'), result_list)

    # 填充空值为0（表示该家庭中无人符合该条件）
    merged_df.fillna(0, inplace=True)

    # 自动生成占比列（分子/家庭规模）
    for col in count_var_names:
        prop_col = col + '占比'
        merged_df[prop_col] = merged_df[col] / merged_df['家庭规模']
    merged_df.fillna(0, inplace=True)

    return merged_df



