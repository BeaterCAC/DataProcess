# @Time    : 2025/1/21 11:09
# @Author  : Aocheng Chen
# @FileName: Visualization.py
# @Describe: 该文件用于数据的可视化，包含变量自身的分布以及两个变量之间的关系

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DataProcess_2025.source.OutlierProcess import detect_outliers
from ipywidgets import (interact, Dropdown, SelectMultiple,
                        RadioButtons, IntSlider, FloatRangeSlider,
                        VBox, HBox, widgets, Button, Output, FloatText)
from IPython.display import display

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来显示负号


def distinguish_variable_types(df, threshold=0.001):
    """
    区分分类变量和数值变量，基于数据类型和唯一值占比。
    """
    categorical_vars = []
    numerical_vars = []

    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_vars.append(col)
        else:
            unique_count = df[col].nunique()
            non_na_count = df[col].count()
            if non_na_count == 0:
                continue
            unique_ratio = unique_count / non_na_count
            if unique_ratio < threshold:
                categorical_vars.append(col)
            else:
                numerical_vars.append(col)

    return categorical_vars, numerical_vars


def plot_variable_distributions(df):
    """
    绘制所选变量的分布图，并提供交互式选择。
    """
    # 调用 distinguish_variable_types 函数区分分类变量和数值变量
    categorical_vars, numerical_vars = distinguish_variable_types(df)
    all_variables = categorical_vars + numerical_vars

    # 创建控件
    variable_widget = SelectMultiple(
        options=all_variables,
        value=all_variables[:1],  # 默认选择第一个变量
        description='变量选择:',
        disabled=False
    )

    max_plots_widget = IntSlider(
        value=5,
        min=1,
        max=len(all_variables),
        step=1,
        description='最大图表数:',
        continuous_update=False
    )

    confirm_button = Button(description="生成图表", button_style='success')
    output_area = widgets.Output()

    # 新增：切换数据状态的按钮和提示
    log_switch_button = Button(description="切换到对数数据", icon='log', button_style='info')
    data_status_text = widgets.Label(value="当前数据：原始数据")
    use_log_data = False  # 初始状态为原始数据

    # 新增：对数处理选项控件
    log_options = RadioButtons(
        options=['忽略零或负值', '添加常数'],
        value='忽略零或负值',
        description='对数处理方式:',
        disabled=True
    )
    constant_input = FloatText(
        value=1,
        description='常数值:',
        disabled=True
    )

    range_sliders = {}  # 存储数值变量对应的滑块

    # 布局控件
    controls = HBox([variable_widget, max_plots_widget, confirm_button])
    display(VBox([controls, log_switch_button, data_status_text, log_options, constant_input, output_area]))

    def get_variable_range(var_name, data=df):
        """获取数值变量的数据范围"""
        return (data[var_name].min(), data[var_name].max())

    def create_sliders(selected_vars, data=df):
        """为数值变量创建范围滑块"""
        for var in selected_vars:
            if var in numerical_vars and var not in range_sliders:
                vmin, vmax = get_variable_range(var, data)
                slider = FloatRangeSlider(
                    value=[vmin, vmax],
                    min=vmin,
                    max=vmax,
                    step=(vmax - vmin)/100,
                    description=f'{var}范围:',
                    readout_format='.2f',
                )
                range_sliders[var] = slider
        return [range_sliders[var] for var in selected_vars if var in numerical_vars]

    def on_confirm_clicked(b):
        """确定按钮点击事件"""
        with output_area:
            output_area.clear_output()
            selected_vars = list(variable_widget.value)[:max_plots_widget.value]
            # 生成滑块需基于处理后的数据范围
            processed_df = df.copy()
            if use_log_data:
                for var in selected_vars:
                    if var in numerical_vars:
                        if log_options.value == '忽略零或负值':
                            processed_df = processed_df[processed_df[var] > 0]
                        elif log_options.value == '添加常数':
                            processed_df[var] += constant_input.value
                        processed_df[var] = np.log(processed_df[var])
            # 检查数据是否为空
            if processed_df.empty:
                print("无有效数据可供绘图")
                return

            # 使用处理后的数据创建滑块
            sliders = create_sliders(selected_vars, processed_df)

            # 生成滑块布局
            if sliders:
                slider_box = VBox(sliders)
                display(slider_box)

            # 绘制图表
            plt.close('all')
            fig, axes = plt.subplots(len(selected_vars), 1,
                                     figsize=(10, 6 * len(selected_vars)))
            if len(selected_vars) == 1:
                axes = [axes]

            # 在绘图时使用processed_df的范围
            for ax, var in zip(axes, selected_vars):
                current_var = f"ln({var})" if (use_log_data and var in numerical_vars) else var
                ax.set_xlabel(current_var)
                if var in categorical_vars:
                    sns.countplot(y=var, data=df, order=df[var].value_counts().index, ax=ax)
                    ax.set_title(f'{var} 计数分布')
                else:
                    # 使用滑块的当前值域
                    xmin, xmax = range_sliders[var].value if var in range_sliders else get_variable_range(var, processed_df)
                    outliers = detect_outliers(processed_df[[var]], method='zscore')
                    outlier_mask = outliers[var]

                    # 更新标题显示对数状态
                    sns.kdeplot(processed_df[var], shade=True, ax=ax, cmap='viridis')  # 修改为渐变色
                    ax.scatter(processed_df[outlier_mask][var], [0] * outlier_mask.sum(),
                               color='red', label='异常值')
                    ax.set_xlim(xmin, xmax)
                    ax.set_title(f'{current_var} 密度分布')  # 更新标题
                    ax.legend()

            plt.tight_layout()
            plt.show()

    def on_log_switch_clicked(b):
        """切换数据状态按钮点击事件"""
        nonlocal use_log_data
        use_log_data = not use_log_data
        if use_log_data:
            log_switch_button.description = "切换到原始数据"
            data_status_text.value = "当前数据：对数数据"
            log_options.disabled = False
            constant_input.disabled = log_options.value == '添加常数'
        else:
            log_switch_button.description = "切换到对数数据"
            data_status_text.value = "当前数据：原始数据"
            log_options.disabled = True
            constant_input.disabled = True

    # 绑定事件
    confirm_button.on_click(on_confirm_clicked)
    log_switch_button.on_click(on_log_switch_clicked)
    log_options.observe(lambda change: setattr(constant_input, 'disabled', change['new'] != '添加常数'), names='value')


def plot_relationships(df, x_var, y_var, plot_type, categorical_vars, numerical_vars, third_var=None, multi_vars=None):
    """
    使用Seaborn可视化变量之间的关系。
    """
    plt.figure(figsize=(10, 6))

    # 新增多变量密度图逻辑
    if plot_type == 'ridge_plot':
        if not multi_vars or len(multi_vars) == 0:
            plt.title("请至少选择一个数值变量")
            plt.show()
            return

        # 优化滑块范围：排除异常值影响
        outliers = detect_outliers(df[multi_vars], method='zscore')
        clean_df = df[~outliers.any(axis=1)]
        if clean_df.empty:
            plt.title("无有效数据可供绘图")
            plt.show()
            return

        global_min = clean_df[multi_vars].min().min()  # 排除异常值后的最小值
        global_max = clean_df[multi_vars].max().max()  # 排除异常值后的最大值

        # 创建滑块控件
        range_slider = FloatRangeSlider(
            value=[global_min, global_max],
            min=global_min,
            max=global_max,
            step=(global_max - global_min) / 100,
            description='值域:',
            readout_format='.2f',
        )
        display(range_slider)

        # 创建确认按钮
        confirm_button = Button(description="绘制山脊图", button_style='success')
        output_area = Output()
        display(confirm_button, output_area)

        def draw_ridge_plot(slider_range):
            """根据给定的值域范围绘制山脊图"""
            fig, axes = plt.subplots(len(multi_vars), 1, figsize=(10, 4 * len(multi_vars)), sharex=True,
                                     gridspec_kw={'hspace': -0.5})
            if len(multi_vars) == 1:
                axes = [axes]
            for ax, var in zip(axes, multi_vars):
                filtered_data = clean_df[var][(clean_df[var] >= slider_range[0]) & (clean_df[var] <= slider_range[1])]
                if not filtered_data.empty:  # 确保有数据可用
                    sns.kdeplot(filtered_data, ax=ax, shade=True, color="#69b3a2", alpha=0.7, bw_adjust=1.5)
                    ax.set_yticks([])
                    ax.set_title(var, fontsize=12)
                else:
                    print(f"变量 {var} 在选定范围内没有数据")
            plt.xlim(slider_range)
            plt.xlabel('值域')
            plt.tight_layout()
            plt.show()

        def on_generate_clicked(_):
            """生成图表按钮点击事件处理"""
            slider_range = range_slider.value  # 获取当前滑块值
            with output_area:
                output_area.clear_output()
                draw_ridge_plot(slider_range)

        # 绑定按钮点击事件
        confirm_button.on_click(on_generate_clicked)

    elif plot_type == 'multi_density':
        if not multi_vars or len(multi_vars) == 0:
            plt.title("请至少选择一个数值变量")
            plt.show()
            return
        outliers = detect_outliers(df[multi_vars], method='zscore')
        clean_df = df[~outliers.any(axis=1)]
        if clean_df.empty:
            plt.title("无有效数据可供绘图")
            plt.show()
            return
        for var in multi_vars:
            sns.kdeplot(clean_df[var], label=var, shade=True, alpha=0.5)
        plt.title(f'多变量密度分布对比（共{len(multi_vars)}个变量）')
        plt.xlabel('值域')
        plt.legend()
        plt.show()
        return

    if x_var and y_var:
        plt.figure(figsize=(10, 6))
        if x_var in categorical_vars and y_var in categorical_vars:
            if plot_type == 'heatmap':
                cross_tab = pd.crosstab(df[x_var], df[y_var])
                sns.heatmap(cross_tab, annot=True, cmap="YlGnBu", fmt='d')
                plt.title(f'{x_var} vs {y_var} (Heatmap)')
            elif plot_type == 'bar':
                sns.countplot(data=df, x=x_var, hue=y_var)
                plt.title(f'{x_var} vs {y_var} (Bar Chart)')
            elif plot_type == 'stacked_bar':
                cross_tab = pd.crosstab(df[x_var], df[y_var]).apply(lambda r: r / r.sum(), axis=1)
                cross_tab.plot(kind='bar', stacked=True, colormap='tab10')
                plt.title(f'{x_var} vs {y_var} (Stacked Bar Chart)')
        elif x_var in numerical_vars and y_var in numerical_vars:
            outliers_x = detect_outliers(df[[x_var]], method='zscore')
            outliers_y = detect_outliers(df[[y_var]], method='zscore')
            outlier_mask = outliers_x[x_var] | outliers_y[y_var]
            if plot_type == 'scatter':
                sns.scatterplot(x=df[~outlier_mask][x_var], y=df[~outlier_mask][y_var], label='Normal')
                sns.scatterplot(x=df[outlier_mask][x_var], y=df[outlier_mask][y_var], color='red', label='Outliers')
                plt.title(f'{x_var} vs {y_var} (Scatter Plot with Outliers)')
            elif plot_type == 'line':
                sns.lineplot(x=df[x_var], y=df[y_var])
                plt.title(f'{x_var} vs {y_var} (Line Plot)')
            elif plot_type == 'bubble' and third_var is not None:
                sizes = df[third_var] * 100  # 调整大小比例
                sns.scatterplot(x=df[~outlier_mask][x_var], y=df[~outlier_mask][y_var], size=sizes[~outlier_mask],
                                sizes=(20, 200), alpha=0.6, legend=False)
                sns.scatterplot(x=df[outlier_mask][x_var], y=df[outlier_mask][y_var], color='red', label='Outliers')
                plt.title(f'{x_var} vs {y_var} (Bubble Plot with {third_var})')
            elif plot_type == 'box':
                sns.boxplot(data=df, x=x_var, y=y_var)
                plt.title(f'{x_var} vs {y_var} (Box Plot)')
            elif plot_type == 'violin':
                sns.violinplot(data=df, x=x_var, y=y_var)
                plt.title(f'{x_var} vs {y_var} (Violin Plot)')
        elif x_var in categorical_vars and y_var in numerical_vars:
            if plot_type == 'bar':
                sns.barplot(data=df, x=x_var, y=y_var)
                plt.title(f'{x_var} vs {y_var} (Bar Chart)')
            elif plot_type == 'stacked_bar':
                cross_tab = pd.crosstab(df[x_var], df[y_var]).apply(lambda r: r / r.sum(), axis=1)
                cross_tab.plot(kind='bar', stacked=True, colormap='tab10')
                plt.title(f'{x_var} vs {y_var} (Stacked Bar Chart)')
            elif plot_type == 'grouped_bar':
                sns.catplot(data=df, x=x_var, y=y_var, kind='bar', height=6, aspect=1.2)
                plt.title(f'{x_var} vs {y_var} (Grouped Bar Chart)')
            elif plot_type == 'box':
                sns.boxplot(data=df, x=x_var, y=y_var)
                plt.title(f'{x_var} vs {y_var} (Box Plot)')
        elif x_var in numerical_vars and y_var in categorical_vars:
            if plot_type == 'box':
                sns.boxplot(data=df, x=y_var, y=x_var)
                plt.title(f'{x_var} vs {y_var} (Box Plot)')
        plt.legend()
        plt.show()


def visualize_variable_relationships(df, categorical_vars, numerical_vars):
    """
    提供交互式选择以可视化变量之间的关系。
    """
    # 创建控件
    x_widget = Dropdown(
        options=categorical_vars + numerical_vars,
        value=categorical_vars[0] if categorical_vars else numerical_vars[0],
        description='X变量:',
        disabled=False
    )
    y_widget = Dropdown(
        options=categorical_vars + numerical_vars,
        value=numerical_vars[0] if numerical_vars else categorical_vars[0],
        description='Y变量:',
        disabled=False
    )
    plot_type_widget = RadioButtons(
        options=['scatter', 'line', 'bubble', 'box', 'violin', 'bar',
                 'stacked_bar', 'grouped_bar', 'heatmap', 'multi_density', 'ridge_plot'],
        value='scatter',
        description='图表类型:'
    )
    third_var_widget = Dropdown(
        options=['None'] + numerical_vars,
        value='None',
        description='第三变量:',
        disabled=True
    )
    multi_var_widget = SelectMultiple(
        options=numerical_vars,  # 确保选项正确加载
        description='多选变量:',
        disabled=True,
        layout={'width': '500px'}
    )
    confirm_button = Button(description="生成图表", button_style='success')
    output_area = widgets.Output()

    # 新增：切换数据状态的按钮和提示
    log_switch_button = Button(description="切换到对数数据", icon='log', button_style='info')
    data_status_text = widgets.Label(value="当前数据：原始数据")
    use_log_data = False  # 初始状态为原始数据

    # 新增：对数处理选项控件
    log_options = RadioButtons(
        options=['忽略零或负值', '添加常数'],
        value='忽略零或负值',
        description='对数处理方式:',
        disabled=True
    )
    constant_input = FloatText(
        value=1,
        description='常数值:',
        disabled=True
    )

    # 控件布局
    main_controls = HBox([x_widget, y_widget, plot_type_widget])
    special_controls = HBox([third_var_widget, multi_var_widget])
    display(VBox([main_controls, special_controls, log_switch_button, data_status_text, log_options, constant_input,
                  confirm_button, output_area]))

    def update_controls(_=None):
        """动态控件状态管理"""
        if plot_type_widget.value in ['multi_density', 'ridge_plot']:
            x_widget.disabled = True
            y_widget.disabled = True
            third_var_widget.disabled = True
            multi_var_widget.disabled = False
            multi_var_widget.options = numerical_vars  # 确保选项正确加载
        elif plot_type_widget.value == 'bubble':
            x_widget.disabled = False
            y_widget.disabled = False
            third_var_widget.disabled = False
            multi_var_widget.disabled = True
        else:
            x_widget.disabled = False
            y_widget.disabled = False
            third_var_widget.disabled = True
            multi_var_widget.disabled = True

    def on_confirm_clicked(_):
        """按钮点击事件处理"""
        with output_area:
            output_area.clear_output()
            # 获取当前参数
            plot_type = plot_type_widget.value
            third_var = third_var_widget.value if third_var_widget.value != 'None' else None
            multi_vars = list(multi_var_widget.value)
            # 输入验证
            if plot_type in ['multi_density', 'ridge_plot'] and not multi_vars:
                print("错误：请至少选择一个数值变量")
                return
            # 根据数据状态选择原始数据或对数数据
            processed_df = df.copy()
            if use_log_data:
                vars_to_process = multi_vars or [x_widget.value, y_widget.value]
                for var in vars_to_process:
                    if var in numerical_vars:
                        if log_options.value == '忽略零或负值':
                            processed_df = processed_df[processed_df[var] > 0]
                        elif log_options.value == '添加常数':
                            processed_df[var] += constant_input.value
                        processed_df[var] = np.log(processed_df[var])

            # 执行绘图
            try:
                if plot_type in ['multi_density', 'ridge_plot']:
                    plot_relationships(processed_df, None, None, plot_type,
                                       categorical_vars, numerical_vars,
                                       multi_vars=multi_vars)
                else:
                    plot_relationships(processed_df, x_widget.value, y_widget.value,
                                       plot_type, categorical_vars, numerical_vars,
                                       third_var)
            except Exception as e:
                print(f"绘图错误: {str(e)}")

    def on_log_switch_clicked(b):
        """切换数据状态按钮点击事件"""
        nonlocal use_log_data
        use_log_data = not use_log_data
        if use_log_data:
            log_switch_button.description = "切换到原始数据"
            data_status_text.value = "当前数据：对数数据"
            log_options.disabled = False
            constant_input.disabled = log_options.value == '添加常数'
        else:
            log_switch_button.description = "切换到对数数据"
            data_status_text.value = "当前数据：原始数据"
            log_options.disabled = True
            constant_input.disabled = True

    # 绑定事件
    plot_type_widget.observe(update_controls, names='value')
    confirm_button.on_click(on_confirm_clicked)
    log_switch_button.on_click(on_log_switch_clicked)
    log_options.observe(lambda change: setattr(constant_input, 'disabled', change['new'] != '添加常数'), names='value')
    update_controls()  # 初始化控件状态
