'''
Author: [hyc]
Description: 
'''

#导入基础库
import pandas as pd
import numpy as np
import warnings

#from regex import E
warnings.filterwarnings("ignore")

# 二元变量换成0-1
def trans_onehot(data, sheet):
    """
    本函数用于将二元变量的12变成10
    :param data: pandas DataFrame，包含需要处理的数据
    :param sheet: excel，包含需要处理的方式
    :return: 处理后的DataFrame
    """
    tmpsheets =sheet
    my_list = []

    tmpsheet = tmpsheets.loc[(tmpsheets['二元转01'] == 1),['新数据名','二元转01']]


    curvar = tmpsheet['新数据名'].tolist()
    hebin = tmpsheet['二元转01'].tolist()
    tmplist = list(zip(curvar,hebin))
    my_list.extend(tmplist)


    df = data.copy()
    for att in curvar:
        df[att] = df[att].replace({1:1,2:0})
    return df


# 多选题变量生成01变量 ，pandas, pd.get_dummies(data,prefix=)的使用,data表示数组、Series或DataFrame, prefix表示给分组加前缀。
def encoder_onehot(data,sheet):
    """
    本函数用于生成01变量，vlist为att包含的值
    :param data: pandas DataFrame，包含需要处理的数据
    :param sheet: excel，包含需要处理的方式
    :return: 处理后的DataFrame
    """
    df = data.copy()

    tmpsheets =sheet
    my_list = []

    tmpsheet = tmpsheets.loc[(pd.notna(tmpsheets['多元转01'])),['新数据名','多元转01']]


    curvar = tmpsheet['新数据名'].tolist()
    hebin = tmpsheet['多元转01'].tolist()
    tmplist = list(zip(curvar,hebin))
    my_list.extend(tmplist)

    for x in my_list:
        att = x[0]
        vlist = eval(x[1])

        names = []
        for i in vlist:
            names.append(f'{att}_bn_{i}')
        print(names)
        var_dict = {x: y for x, y in zip(names, vlist)}
        zero_Matrix = np.zeros((len(df[att]),len(names)))
        var_dummies = pd.DataFrame(zero_Matrix, columns = names)
        tmp_list = [i.split('-') for i in df[att]]
        for val in var_dict.keys():
            var_dummies[val] = [1 if var_dict[val] in i else 0 for i in tmp_list]
        df = pd.concat([df,var_dummies],axis=1)
    return df
