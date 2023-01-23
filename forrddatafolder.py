#This program is for merging documents in a folder

import numpy as np
import pandas as pd
import glob
import os

os.chdir("YOUR PATH") 


csv_list = glob.glob('*.csv')
print('发现%s个CSV文件'% len(csv_list))
print('对文件进行处理中')
for i in csv_list:
    fr = open(i,'r',encoding='utf-8').read()
    with open('M7test.csv','a',encoding='utf-8') as f:
        f.write(fr)
print('所有文件合并完成！')    #完成后在这个路径的合并文件需要删除第一列才能识别16channels
