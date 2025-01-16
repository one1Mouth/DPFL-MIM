import os
import re
import matplotlib.pyplot as plt
import numpy as np
import h5py
from openpyxl import Workbook

# 预设10种颜色
COLORS = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄绿色
    '#17becf',  # 青色
]

# 预设 4 种线型（只有这4种）
LINE_STYLES = [
    '-',  # 实线
    '--',  # 虚线
    # '-.',  # 点划线
    ':',  # 点线
]


def getH5FilesName(folder_path='./h5files'):
    # 创建一个空列表来存储文件名
    file_names = []
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 检查是否是文件
        if os.path.isfile(os.path.join(folder_path, filename)) and filename.endswith('.h5'):
            file_names.append(filename)  # 添加文件名到列表中

    # 使用自定义的排序函数
    def sort_key(filename):
        # 使用正则表达式提取"_0"之前的数字
        match = re.search(r'(\d+)_0\.h5', filename)
        if match:
            return int(match.group(1))  # 返回匹配的数字
        return 0  # 如果没有找到数字，返回0

    # 对文件名列表进行排序
    file_names.sort(key=sort_key)

    # print(file_names)  # 打印文件名列表
    return file_names


def getDataDict(folder_path='./h5files'):
    """
    这个函数目的为：读取folder_path中file_names_list中所有文件的:
     train_loss_list, test_loss_list, personal_avg_test_acc, server_test_acc
     并封装到一个字典中，然后返回，方便后续作图
    """
    file_names_list = getH5FilesName(folder_path)
    value_list_name = ["train_loss", "test_loss", "personal_avg_test_acc", "server_test_acc"]
    son_dict = {key: [] for key in value_list_name}  # 做一个子dict出来
    data_dict = {key: son_dict.copy() for key in file_names_list}  # 初始化key值为文件名且value值为son_dict
    # 靠，上一行没.copy()导致这个key是个指针，后面循环全部变成最后一个数据
    for file_name in file_names_list:
        with h5py.File(folder_path + "\\" + file_name, 'r') as hf:
            # 封装数据
            data_dict[file_name]["train_loss"] = hf['rs_train_loss'][:]
            data_dict[file_name]["test_loss"] = hf['rs_server_loss'][:]
            data_dict[file_name]["personal_avg_test_acc"] = hf['rs_test_acc'][:]
            data_dict[file_name]["server_test_acc"] = hf['rs_server_acc'][:]
            data_dict[file_name]["epsilon_list"] = hf['epsilon_list'][:]
    return file_names_list, data_dict


def find_first_below_threshold(lst, threshold):
    """
    在Python中，你可以使用列表推导（list comprehension）结合next函数和生成器表达式来实现这个功能。
    下面是一个函数，它接受一个列表和一个阈值，然后返回列表中第一个低于该阈值的元素的索引。
    如果没有这样的元素，则抛出异常。
    """
    try:
        return next(i for i, x in enumerate(lst) if x >= threshold)
    except StopIteration:
        raise StopIteration("Threshold is too big.")  # 抛出自定义异常


def find_first_below_threshold_process_nan(lst, threshold):
    """
    在Python中，你可以使用列表推导（list comprehension）结合next函数和生成器表达式来实现这个功能。
    下面是一个函数，它接受一个列表和一个阈值，然后返回列表中第一个低于该阈值的元素的索引。
    如果没有这样的元素，则返回-1。
    """
    try:
        return next(i for i, x in enumerate(lst) if x >= threshold)
    except StopIteration:
        return -1


def find_first_above_threshold(lst, threshold):
    """
    在Python中，你可以使用列表推导（list comprehension）结合next函数和生成器表达式来实现这个功能。
    下面是一个函数，它接受一个列表和一个阈值，然后返回列表中第一个高于该阈值的元素的索引。
    如果没有这样的元素，则抛出异常。
    """
    try:
        return next(i for i, x in enumerate(lst) if x <= threshold)
    except StopIteration:
        raise StopIteration("Threshold is too small.")  # 抛出自定义异常


def find_first_above_threshold_process_nan(lst, threshold):
    """
    在Python中，你可以使用列表推导（list comprehension）结合next函数和生成器表达式来实现这个功能。
    下面是一个函数，它接受一个列表和一个阈值，然后返回列表中第一个高于该阈值的元素的索引。
    如果没有这样的元素，则抛出异常。
    """
    try:
        return next(i for i, x in enumerate(lst) if x <= threshold)
    except StopIteration:
        return -1


def draw_multiple_line(x_axis, data, legends, title, x_label, y_label, pdf_name=None, save_as_pdf=False):
    """
    将二维list data绘制成包含多条线的折线图
    :param x_axis: 横坐标
    :param data: 二维列表，其中每个子列表代表一条线的数据
    :param legends: 这些折线的图例
    :param title: 折线图的标题
    :param x_label: x轴的标签
    :param y_label: y轴的标签
    :param save_as_pdf: 是否保存为PDF文件，默认为False
    """
    # 创建图形和轴对象
    fig, ax = plt.subplots()

    # 绘制每一条线
    for i, dataset in enumerate(data):
        # 假设每个子列表的第一个元素是x值，其余元素是y值
        color = COLORS[i % len(COLORS)]  # 循环使用颜色列表
        line_style = LINE_STYLES[i % len(LINE_STYLES)]  # 循环使用线型列表
        ax.plot(x_axis, dataset, label=legends[i], linestyle=line_style, color=color)

    # 设置图例
    ax.legend()

    # 设置标题和轴标签
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # 显示网格
    ax.grid(True)

    # 如果需要，保存为PDF文件
    if save_as_pdf:
        plt.savefig(pdf_name, format='pdf', bbox_inches='tight')

    # 显示图形
    plt.show()


def dict_to_lists(file_names_list, data_dict):
    train_loss_lists = []
    server_loss_lists = []
    personal_test_acc_lists = []
    server_acc_lists = []
    epsilon_lists = []
    for file_name in file_names_list:
        train_loss_lists.append(data_dict[file_name]["train_loss"])
        server_loss_lists.append(data_dict[file_name]["test_loss"])
        personal_test_acc_lists.append(data_dict[file_name]["personal_avg_test_acc"])
        server_acc_lists.append(data_dict[file_name]["server_test_acc"])
        epsilon_lists.append(data_dict[file_name]["epsilon_list"])
    return train_loss_lists, server_loss_lists, personal_test_acc_lists, server_acc_lists, epsilon_lists


def create_excel_with_sheets(K, train_loss_lists, server_loss_lists, personal_test_acc_lists, server_acc_lists,
                             epsilon_lists, save_name="default.xlsx"):
    # 创建一个工作簿
    wb = Workbook()
    wb.remove(wb.active)  # 移除默认创建的第一个工作表

    # 创建并命名工作表
    sheets = {
        'train_loss_lists': train_loss_lists,
        'server_loss_lists': server_loss_lists,
        'personal_test_acc_lists': personal_test_acc_lists,
        'server_acc_lists': server_acc_lists,
        'epsilon_lists': epsilon_lists
    }

    for sheet_name, data in sheets.items():
        ws = wb.create_sheet(title=sheet_name)
        # 添加列标题
        ws.append(['T'] + [str(i) for i in range(0, 501)])
        # 写入数据到工作表
        for i, item in enumerate(data):
            # 将NaN值转换为字符串"NaN"
            item_with_nan_as_str = [f'K={K[i % len(K)]}'] + [str(x) if np.isnan(x) else x for x in item]
            ws.append(item_with_nan_as_str)

    # 保存工作簿
    wb.save(save_name)
    print(save_name + "文件已创建，并包含五个工作表")


def dict_to_xlsx(K, file_names_list, data_dict, output_file_name='result.xlsx'):
    train_loss_lists, server_loss_lists, personal_test_acc_lists, server_acc_lists, epsilon_lists = dict_to_lists(
        file_names_list, data_dict)
    create_excel_with_sheets(K, train_loss_lists, server_loss_lists, personal_test_acc_lists, server_acc_lists,
                             epsilon_lists, save_name=output_file_name)


if __name__ == '__main__':
    folder_path = './h5files'  # 你的文件夹路径
    file_names_list, data_dict = getDataDict(folder_path)
    # K = [1, 2, 3, 5, 7, 10, 15, 20, 35, 50, 100, 150, 200, 300, 500]
    # dict_to_xlsx(K, file_names_list, data_dict, output_file_name="0905_MNIST_Dir(0.1)_T=500.xlsx")
    K = [1, 3, 5, 10, 20, 50]
    dict_to_xlsx(K, file_names_list, data_dict, output_file_name="0907_MNIST_Dir(0.1)_T=500_diff_sigma.xlsx")
