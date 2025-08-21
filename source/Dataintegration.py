'''
Author: [QJX, jessie@swufe.edu.cn]
Date: 2025-01-16 10:14:36
Description: 
    1. 数据集成
    本代码仅考虑表格之间横向拼接。
    多个数据通过关键词集成, 可通过调用类或函数实现:
        1) 类: class data_integration() 
        2) 函数: def data_integ()
    2. 对数据进行了变量重命名
        1) def rename_columns(data,sheets)
    3. 对特殊数据的处理,如生日日期变换成年龄等
'''


import pandas as pd
import numpy as np
import warnings
import copy

warnings.filterwarnings("ignore")

# 数据整合_类
# 整合的数据类型: datalist,数据列表（list）,每个元素为DataFrame

class DataIntegration:
    def __init__(self, method='merge', on='id', how='inner', merge_type=0, ndata_id=None, ndata_key=None, drop_dlist=False):
        """
        初始化数据整合类。
        :param method: str, 数据拼接方法，'merge'(基于键值的合并操作) 或 'concat'（多个对象行/列堆叠）。
        :param on: str或list, 拼接方式为'merge'时的键值, 表每个数据的唯一标识符。
                不同数据集关系为1 vs 1时:
                    1) str,链接列名相同
                    2) 一维list,链接列名不同,给出各表格链接的列名,['id','hhid']
                不同数据集关系为1 vs n时,需要对1进行拆分或者对n多个字段进行拼接(未完成):
                    3) 对1字段进行拆分处理 
                    4) 对n多个字段进行拼接
        :param how: str, 数据拼接方式，与 merge 中的 how 参数一致, 默认为'inner'
        :param merge_type: int, 数据拼接的方式, 0 表示1对1, 1 表示n对1。
                        0: 默认, 表格之间是1对1的关系; 
                        1: 表格之间存在n对1的关系,根据on按1生成数据;
                        2: 表格之间存在n对1的关系,根据on按n生成数据.
        :param ndata_id: dict, 默认为None, 当merge_type=1时,指定需要确定n对1关系中 n 数据集取值的样本 id 列名和值
                        例如ndata_id={'hhead':1}:表示只取字段hhead为1的样本
        :param ndata_key: dict,当存在n对1关系, 对于n的数据需要拼接多个字段确定n数据样本唯一id
                        例如ndata_key={0:['hhid','pline'],1:['hhid','pline']},0
        :param drop_dlist: bool,是否删除 datalist 中的重复列。
        """
        self.method = self._validate_method(method)
        self.on = self._validate_on(on)
        self.how = how
        self.merge_type = self._validate_merge_type(merge_type)
        self.ndata_id = ndata_id
        self.ndata_key = ndata_key
        self.drop_dlist = drop_dlist
        self.datalist = None
        self.keyid = None

    def _validate_method(self, method):
        if method not in ['merge', 'concat']:
            raise ValueError("method 参数必须是 'merge' 或 'concat'")
        return method

    def _validate_on(self, on):
        if not isinstance(on, (str, list)):
            raise ValueError("on 参数必须是字符串或列表")
        return on

    def _validate_merge_type(self, merge_type):
        if merge_type not in [0, 1]:
            raise ValueError("merge_type 参数必须是 0 或 1")
        return merge_type

    def key_id(self):
        """
        将连接关键词字段统一命名，对不同的关键词统一命名为 key_id。
        """
        if isinstance(self.on, str):
            self.keyid = self.on
        elif isinstance(self.on, list):
            for i, col in enumerate(self.on):
                self.datalist[i].rename(columns={col: 'key_id'}, inplace=True)
            self.keyid = 'key_id'
        else:
            raise ValueError("on 参数类型错误，请输入字符串或列表")

    def unique_id(self):
        """
        将存在多个字段确定唯一样本 id 的生成唯一样本 id 'unique_id'。
        """
        for idx, cols in self.ndata_key.items():
            self.datalist[idx]['unique_id'] = self.datalist[idx][cols].astype('str').agg('_'.join, axis=1)

    def drop_duplicateslist(self):
        """
        删除各表格中重复字段。
        """
        first_columns = self.datalist[0].columns
        for i in range(1, len(self.datalist)):
            droplist = [x for x in self.datalist[i].columns.intersection(first_columns) if x != self.keyid]
            self.datalist[i].drop(columns=droplist, inplace=True)

    def data_combine(self):
        """
        拼接数据。
        """
        dataset = self.datalist[0]
        for df in self.datalist[1:]:
            dataset = pd.merge(dataset, df, on=self.keyid, how=self.how)
        if self.merge_type == 1:  # 选出n对1中n数据集的样本
            for col, val in self.ndata_id.items():
                dataset = dataset[dataset[col] == val]
        return dataset

    def fit(self, dflist):
        """
        执行数据整合。
        :param dflist: 数据列表，每个元素为 DataFrame。
        :return: 整合后的 DataFrame。
        """
        self.datalist = dflist
        if self.on == 'unique_id' or self.ndata_key:
            self.unique_id()
        self.key_id()
        if self.drop_dlist:
            self.drop_duplicateslist()
        return self.data_combine()
    

# 数据整合_函数
def data_integration(datalist,on='id',merge_type=0,ndata_id=False,drop_dlist=False):
    """
    datalist: list,  数据列表,每个元素为DataFrame
    on: str or list, 每个数据的唯一标识符
        不同数据集关系为1 vs 1时:
            1) str,链接列名相同
            2) 一维list,链接列名不同,给出各表格链接的列名,['id','hhid']
        不同数据集关系为1 vs n时:
            3) 二维list,链接列名不同,
            如hh与ind的数据关系是1对多,ind每个样本的id由家庭的'hhid'和家庭成员编号'pline'变量,
            这时需要输入[['hhid'],['hhid','pline']]
    merge_type: int,数据拼接的方式
            0: 默认, 表格之间是1对1的关系; 
            1: 表格之间存在n对1的关系,根据on按1生成数据;
            2: 表格之间存在n对1的关系,根据on按n生成数据.
    ndata_id: dict, 当merge_type=1时,指定需要确定n对1关系中n数据集的样本id列名和值,默认为False,即不指定
    drop_dlist: bool, 是否删除datalist中的重复的列,默认:False
    """

    # （1）若存在1vn的数据关系,生成唯一变量unique_id
    if type(on) == str:
        keyid = on
    elif np.ndim(on) == 1:  # 统一关键字名字
        for i in range(len(on)): 
            datalist[i].rename(columns={on[i]:'unique_id'},replace=True)
        keyid = 'unique_id'
    else:  # 将存在多个字段确定id的表格生成唯一的'unique_id'
        common_id = set(on[0])  # 从第一个子列表开始
        for subid in on[1:]:  # 遍历剩余的子列表
            common_id &= set(subid)  #
        keyid = list(common_id)
        # 针对多个字段确定样本唯一性的重新生成关键词字段'unique_id'
        for i in range(len(datalist)):  # 遍历 datalist 中的每个 DataFrame
            if len(on[i]) == 1:  # 如果只有一个字段，则跳过
                continue
            else:  # 如果有多个字段，则将它们连接起来生成唯一 ID
                # 使用 apply 函数来生成唯一 ID，这样可以处理任意数量的字段
                datalist[i]['unique_id'] = datalist[i][on[i]].astype('str').apply(lambda x: '_'.join(x), axis=1)
    # 删除各表格中重复字段
    if drop_dlist:
        # 找出相同的字段,后面每一个dataframe与第一个的dataframe的重复字段
        first_columns = datalist[0].columns
        for i in range(1, len(datalist)):
            # 找出与第一个DataFrame相同的字段，但排除keyid
            droplist = [x for x in datalist[i].columns.intersection(first_columns) if x != keyid]
            # 删除这些字段
            datalist[i] = datalist[i].drop(columns=droplist)

    # 拼接数据
    dataset = datalist[0]
    for df in datalist[1:]:
        dataset = pd.merge(dataset, df, on=keyid)
    if merge_type == 1:   # 选出n对1中n数据集的样本
        for i,j in ndata_id.items():
            dataset = dataset[dataset[i]==j]

    # 保存文件
    return dataset


# 变量批量重新命名,并截取需要的变量数据
def rename_columns(data,sheets,drop_var=False):
    """
        变量数据重新命名。
        :param data: 输入数据，为 DataFrame。
        :param sheets: 输入需处理的表格，为 DataFrame。
        :param drop_var: Bool, 返回data所有数据为False, 返回data中重新命名的数据为True。
        :return: 重新命名的 DataFrame。
    """

    df = data.copy()
    tmpsheet = sheets.loc[(pd.notna(sheets['原始数据名'])),['原始数据名','新数据名']]
    var_name_old = tmpsheet['原始数据名'].tolist()
    var_name_new = tmpsheet['新数据名'].tolist()
    dict_temp = {i:j for i,j in zip(var_name_old,var_name_new)}
    # 更换变量名
    df.rename(columns=dict_temp,inplace=True)
    print(f'重命名完成')
    
    if drop_var == False:
        return df
    else:
        df_res = df.loc[:,var_name_new]
        return df_res # 截取需要的变量,即重新命名的变量

# 数据的基本转换
def data_base_transfer(data, String_value):

    df = data.copy()
    # 替换所有'.d','.r','.e','.n'
    df.replace(['.d','.r','.e','.n'],np.nan,inplace=True)

    # 把所有数值型变量先统一变成float类型,字符串转为纯字符串
    name = df.columns.tolist()
    Numerical_value = [x for x in name if x not in String_value and x != 'hhid']
    for i in Numerical_value:
        df[i] = df[i].apply(lambda x:float(x) if not pd.isnull(x) else x)
    for i in String_value:
        df[i] = df[i].apply(lambda x:str(x) if not pd.isnull(x) else x)
    
    return df