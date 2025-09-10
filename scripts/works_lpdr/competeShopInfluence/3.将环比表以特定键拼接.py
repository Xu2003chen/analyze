import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取data/final_data下所以csv文件，新增一列，列名为文件名称，然后拼接到一个表中保存
import pandas as pd

files = os.listdir('data/final_data')
files = [file for file in files if file.endswith('.csv')]

dfs = []
for file in files:
    df = pd.read_csv(f'data/final_data/{file}')
    df['时间段标识'] = file
    dfs.append(df)

result = pd.concat(dfs, ignore_index=True)
result.to_csv('竞争门店开业后数据.csv', index=False)

