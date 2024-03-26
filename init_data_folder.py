import os
import stat
import time

"""
此程序用于生成source下的DATA_tem全空文件夹
包括0（timber&walk），1（timber&walk），2（timber&walk），npy文件夹
"""


# 移除含有文件的目录
def rmdir(rm_path):
    if os.path.exists(rm_path):
        if os.path.isdir(rm_path):
            # （先清空目录下面的文件），最后再删除根目录
            for root, dirs, files in os.walk(rm_path, topdown=False):
                for name in files:
                    filename = os.path.join(root, name)
                    os.chmod(filename, stat.S_IWUSR)
                    os.remove(filename)
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            time.sleep(0.1)
            os.rmdir(rm_path)


# 添加存放不同分类的数据文件夹
def add_class_folder(add_class_path):
    path1 = add_class_path + '\\timber'
    path2 = add_class_path + '\\walk'
    if not os.path.exists(path1):
        os.mkdir(path1)
    if not os.path.exists(path2):
        os.mkdir(path2)


# 初始化文件夹
def init_data_folder(data_path):
    data_path0 = data_path + '\\npy'
    data_path1 = data_path + '\\0'
    data_path2 = data_path + '\\1'
    data_path3 = data_path + '\\2'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        rmdir(data_path)
        os.mkdir(data_path)
    os.mkdir(data_path0)
    if not os.path.exists(data_path1):
        os.mkdir(data_path1)
        add_class_folder(data_path1)
    if not os.path.exists(data_path2):
        os.mkdir(data_path2)
        add_class_folder(data_path2)
    if not os.path.exists(data_path3):
        os.mkdir(data_path3)
        add_class_folder(data_path3)
    print('initial data folder success!')


if __name__ == '__main__':
    path = r'source\DATA_tem'
    init_data_folder(path)
