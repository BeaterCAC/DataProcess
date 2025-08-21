# @Time    : 2025/3/22 21:16
# @Author  : Aocheng Chen
# @FileName: BasicProcess.py
# @Describe:


import pandas as pd
import numpy as np
import warnings
import copy

warnings.filterwarnings("ignore")


def myfunction():
    print("Hello, World!")


def get_all_data(hh_data, ind_data, master_data):
    hh = hh_data
    ind = ind_data
    master = master_data

    # （1）以hhid和pline生成唯一变量unique_id，代表个人
    master['unique_id'] = master['hhid'].astype('str') + "_" + master['pline'].astype('str')
    ind['unique_id'] = ind['hhid'].astype('str') + "_" + ind['pline'].astype('str')

    # 去掉可能会干扰的列
    droptist1 = [x for x in ind.columns.intersection(master.columns).to_list() if
                 x != 'unique_id']  # 不同表格存在相同变量名（如都有hhid_2013）则报错，找出相同列表
    master1 = master.drop(columns=droptist1)

    # 拼接ind和master
    ind_master = pd.merge(ind, master1, on='unique_id', suffixes=('', ''))

    # 选出hhead为1的户主样本
    ind_master_h = ind_master[ind_master['hhead'] == 1]

    # (2)将ind和master合并的表格与hh合并，合并前删除hh与之重复字段
    droptist2 = [x for x in ind_master_h.columns.intersection(hh.columns).to_list() if x != 'hhid']
    ind_master_h1 = ind_master_h.drop(columns=droptist2)
    ind_master_hh = pd.merge(hh, ind_master_h1, on='hhid', suffixes=('', ''))

    # 保存文件
    return ind_master_hh


def rename_columns(alldata, sheets):
    ind_master_hh = alldata
    # df = ind_master_hh.query('a2001 in [1,2,3,6]').copy()  #只要本人（22735）、配偶（9169）的样本
    df = ind_master_hh.copy()

    # 获取sheet中所有的题号和简称
    tmpsheets = sheets
    rename_list = []

    tmpsheet = tmpsheets.loc[(pd.notna(tmpsheets['原始数据名'])), ['原始数据名', '新数据名']]
    var_name_old = tmpsheet['原始数据名'].tolist()
    var_name_new = tmpsheet['新数据名'].tolist()
    dict_temp = {i: j for i, j in zip(var_name_old, var_name_new)}
    rename_list.extend(var_name_new)

    # 更换变量名
    df.rename(columns=dict_temp, inplace=True)
    print(f'重命名完成')

    # 截取需要的变量
    df_res = df.loc[:, rename_list]

    return df_res


def base_processing(data, String_value):
    df = data.copy()
    # 替换所有'.d','.r','.e','.n'
    df.replace(['.d', '.r', '.e', '.n'], np.nan, inplace=True)

    # 把所有数值型变量先统一变成float类型,字符串转为纯字符串
    name = df.columns.tolist()
    Numerical_value = [x for x in name if x not in String_value and x != 'hhid']
    for i in Numerical_value:
        df[i] = df[i].apply(lambda x: float(x) if not pd.isnull(x) else x)
    for i in String_value:
        df[i] = df[i].apply(lambda x: str(x) if not pd.isnull(x) else x)

    return df