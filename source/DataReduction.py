# @Time    : 2025/2/17 11:47
# @Author  : Aocheng Chen
# @FileName: DataReduction.py
# @Describe: 本文件用于数据规约，包含PCA、因子分析等功能，提供交互式界面。

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from DataProcess_2025.source.Visualization import distinguish_variable_types
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity


def pca_analysis(df):
    # 选择数值型列
    _, numeric_cols = distinguish_variable_types(df)

    # 创建交互组件
    col_selector = widgets.SelectMultiple(
        options=numeric_cols,
        description='选择特征',
        layout={'width': '500px'},
        rows=10
    )

    dim_input = widgets.IntText(
        value=2,
        min=1,
        max=len(numeric_cols),
        step=1,
        description='降维维度:',
        disabled=False,
        layout={'width': '300px'}
    )

    # 新增排序组件
    sort_pc_dropdown = widgets.Dropdown(
        options=['PC1'],
        value='PC1',
        description='排序依据:',
        disabled=False
    )

    # 维度变化时更新排序选项
    def update_sort_options(change):
        new_dim = change['new']
        sort_pc_dropdown.options = [f'PC{i + 1}' for i in range(new_dim)]
        sort_pc_dropdown.value = f'PC{min(new_dim, len(sort_pc_dropdown.options))}'

    dim_input.observe(update_sort_options, names='value')

    run_button = widgets.Button(description='执行PCA分析', button_style='success')
    output = widgets.Output()

    def on_run_button_clicked(b):
        with output:
            clear_output(wait=True)
            display(HTML("<p style='color: blue;'>正在生成中...</p>"))

            selected_cols = list(col_selector.value)
            n_components = dim_input.value
            sort_by = sort_pc_dropdown.value

            if not selected_cols:
                print("错误：请至少选择一个数值型列！")
                return

            if n_components > len(selected_cols):
                print(f"错误：降维维度不能大于所选特征数量({len(selected_cols)})！")
                return

            # 数据预处理
            data = df[selected_cols].dropna()
            if len(data) < 2:
                print("错误：有效数据不足！")
                return

            # 标准化数据
            scaler = StandardScaler()
            X_std = scaler.fit_transform(data)

            # 执行PCA
            pca = PCA(n_components=n_components)
            pca.fit(X_std)
            scores = pca.transform(X_std)
            explained_ratio = pca.explained_variance_ratio_

            # 计算结果
            explained_variance = pca.explained_variance_
            components = pca.components_

            # 计算载荷矩阵
            loadings = components.T * np.sqrt(pca.explained_variance_)

            # 创建结果DataFrame
            variance_df = pd.DataFrame({
                '主成分': [f'PC{i + 1}' for i in range(len(explained_variance))],
                '解释方差': explained_variance,
                '方差比例': explained_ratio,
                '累计方差比例': np.cumsum(explained_ratio)
            })

            # 显示结果
            print("主成分分析结果摘要：")
            display(variance_df.set_index('主成分'))

            # 创建载荷矩阵DataFrame
            loadings_df = pd.DataFrame(
                loadings,
                index=selected_cols,
                columns=[f'PC{i + 1}' for i in range(loadings.shape[1])]
            ).sort_values(by=sort_by, ascending=True)  # 修改排序方式

            # 可视化部分
            plt.figure(figsize=(18, 12))

            # 方差解释图
            plt.subplot(2, 2, 1)
            sns.barplot(x=variance_df['主成分'], y=variance_df['方差比例'], palette='Blues_d')
            plt.title('各主成分解释方差比例')
            plt.axhline(0.8, color='red', linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)

            # 累计方差图
            plt.subplot(2, 2, 2)
            sns.lineplot(x=variance_df['主成分'], y=variance_df['累计方差比例'],
                         marker='o', color='orange')
            plt.axhline(0.8, color='red', linestyle='--', alpha=0.5)
            plt.title('累计解释方差比例')
            plt.xticks(rotation=45)

            # 载荷热力图
            plt.subplot(2, 2, 3)
            sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0,
                        fmt=".2f", linewidths=0.5)
            plt.title(f'主成分载荷矩阵（按{sort_by}排序）')
            plt.xlabel('主成分')

            # 主成分得分分布图
            plt.subplot(2, 2, 4)
            df_scores = pd.DataFrame(scores, columns=[f'PC{i + 1}' for i in range(n_components)])
            sns.pairplot(df_scores, plot_kws={'alpha': 0.6})
            plt.suptitle('主成分得分分布', y=1.02)

            plt.tight_layout()
            plt.show()


    # 绑定事件
    run_button.on_click(on_run_button_clicked)

    # 显示组件
    display(widgets.VBox([
        widgets.HTML("<h3>PCA分析参数设置</h3>"),
        col_selector,
        dim_input,
        sort_pc_dropdown,
        run_button,
        output
    ]))


def factor_analysis(df):
    # 选择数值型列
    _, numeric_cols = distinguish_variable_types(df)

    # 创建交互组件
    col_selector = widgets.SelectMultiple(
        options=numeric_cols,
        description='选择特征',
        layout={'width': '500px'},
        rows=10
    )

    rotation_method = widgets.Dropdown(
        options=[('无旋转', None), ('方差最大化', 'varimax'), ('斜交旋转', 'promax')],
        value=None,
        description='旋转方法:'
    )

    extraction_method = widgets.Dropdown(
        options=[('主成分法', 'principal'), ('最大似然法', 'ml')],
        value='principal',
        description='提取方法:'
    )

    n_factors_input = widgets.IntText(
        value=1,
        min=1,
        max=len(numeric_cols) - 1,
        description='因子数:',
        disabled=False
    )

    # 新增排序组件
    sort_factor_dropdown = widgets.Dropdown(
        options=['因子1'],
        value='因子1',
        description='排序依据:',
        disabled=False
    )

    # 因子数变化时更新排序选项
    def update_factor_options(change):
        new_factor = change['new']
        sort_factor_dropdown.options = [f'因子{i + 1}' for i in range(new_factor)]
        sort_factor_dropdown.value = f'因子{min(new_factor, len(sort_factor_dropdown.options))}'

    n_factors_input.observe(update_factor_options, names='value')

    run_button = widgets.Button(description='执行因子分析', button_style='success')
    output = widgets.Output()

    def run_analysis(b):
        with output:
            clear_output(wait=True)
            display(HTML("<p style='color: blue;'>正在分析中...</p>"))

            selected_cols = list(col_selector.value)
            if len(selected_cols) < 2:
                print("错误：至少需要选择2个变量")
                return

            # 数据预处理
            data = df[selected_cols].dropna()
            if len(data) < 3:
                print("错误：有效样本量不足")
                return

            # 执行检验
            kmo_all, kmo_model = calculate_kmo(data)
            bartlett_stat, bartlett_p = calculate_bartlett_sphericity(data)

            # 显示检验结果
            display(HTML("<h4>数据适切性检验</h4>"))
            display(pd.DataFrame({
                'KMO值': [kmo_model],
                'Bartlett检验P值': [bartlett_p]
            }, index=['结果']).T)

            if kmo_model < 0.5:
                print("警告：KMO值过低（<0.5），数据不适合因子分析")
                return
            if bartlett_p > 0.05:
                print("警告：Bartlett检验不显著（p>0.05），数据不适合因子分析")
                return

            # 自动确定因子数
            fa = FactorAnalyzer(rotation=None, method=extraction_method.value)
            fa.fit(data)
            ev, v = fa.get_eigenvalues()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.scatter(range(1, data.shape[1] + 1), ev)
            plt.plot(range(1, data.shape[1] + 1), ev)
            plt.title('碎石图')
            plt.xlabel('因子数')
            plt.ylabel('特征值')
            plt.axhline(y=1, color='r', linestyle='--')

            auto_factors = sum(ev > 1)
            plt.subplot(1, 2, 2)
            plt.bar(range(len(ev)), ev)
            plt.title('特征值分布')
            plt.axhline(y=1, color='r', linestyle='--')
            plt.show()

            print(f"建议因子数（特征值>1）: {auto_factors}")

            # 执行因子分析
            fa = FactorAnalyzer(
                n_factors=n_factors_input.value,
                rotation=rotation_method.value,
                method=extraction_method.value
            )
            fa.fit(data)

            # 结果展示
            display(HTML("<h4>因子分析结果</h4>"))

            # 公因子方差
            communalities = pd.DataFrame({
                '变量': selected_cols,
                '公因子方差': fa.get_communalities()
            })
            print("\n公因子方差（变量被解释程度）：")
            display(communalities)

            # 因子载荷矩阵
            loadings = pd.DataFrame(
                fa.loadings_,
                index=selected_cols,
                columns=[f'因子{i + 1}' for i in range(n_factors_input.value)]
            ).sort_values(by=sort_factor_dropdown.value, ascending=True)  # 修改排序方式

            print("\n因子载荷矩阵：")
            plt.figure(figsize=(10, 6))
            sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, fmt=".2f")
            plt.title(f'因子载荷热力图（按{sort_factor_dropdown.value}排序）')
            plt.show()

            # 方差解释
            variance_df = pd.DataFrame({
                '因子': [f'因子{i + 1}' for i in range(n_factors_input.value)],
                '解释方差': fa.get_factor_variance()[0],
                '方差比例': fa.get_factor_variance()[1],
                '累计方差比例': fa.get_factor_variance()[2]
            })
            print("\n方差解释表：")
            display(variance_df.set_index('因子'))

            # 因子得分
            scores = fa.transform(data)
            scores_df = pd.DataFrame(
                scores,
                columns=[f'因子{i + 1}得分' for i in range(n_factors_input.value)]
            )
            print("\n前5个样本的因子得分：")
            display(scores_df.head())

            # 旋转矩阵（如果使用斜交旋转）
            if rotation_method.value == 'promax':
                print("\n因子相关矩阵：")
                display(pd.DataFrame(fa.phi_, columns=loadings.columns, index=loadings.columns))

    # 组件交互
    run_button.on_click(run_analysis)

    # 界面布局
    display(widgets.VBox([
        widgets.HTML("<h3>因子分析参数设置</h3>"),
        col_selector,
        widgets.HBox([extraction_method, rotation_method]),
        widgets.HBox([n_factors_input, sort_factor_dropdown]),
        run_button,
        output
    ]))