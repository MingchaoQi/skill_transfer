#批量修改爬取图片格式和文件名，默认操作为将图片按1，2，3，，，顺序重命名

import os

path_in = 'D:/PYTHON/plane_1/su-57'  #待批量重命名的文件夹
class_name = ".jpg"  #重命名后的文件名后缀

file_in = os.listdir(path_in)  #返回文件夹包含的所有文件名
num_file_in = len(file_in)  #获取文件数目
print(file_in, num_file_in)  #输出修改前的文件名

for i in range(0, num_file_in):
    t = str(i + 1)
    new_name = os.rename(path_in + "/" + file_in[i],
                         path_in + "/" + t + class_name)
    #重命名文件名

file_out = os.listdir(path_in)
print(file_out)  #输出修改后的结果
