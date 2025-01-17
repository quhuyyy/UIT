import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
# sns.reset_orig() 
from datetime import datetime, timedelta
import time
df1 = pd.read_csv('222-dataset-v2.csv')
df2 = pd.read_excel('222-labFinal.xlsx')

lab1 = df1 [df1 ['labNum'] == 1]
lab2 = df1 [df1 ['labNum'] == 2]
lab3 = df1 [df1 ['labNum'] == 3]

lab1.reset_index(drop=True, inplace=True)
lab2.reset_index(drop=True, inplace=True)
lab3.reset_index(drop=True, inplace=True)
lab1 = lab1[['studentID','Prelab-result', 'Inlab-result']]
lab2 = lab2[['studentID','Prelab-result', 'Inlab-result']]
lab3 = lab3[['studentID','Prelab-result', 'Inlab-result']]
lab1 = lab1.rename(columns={'Prelab-result': 'Prelab-result1', 'Inlab-result': 'Inlab-result1'})
lab2 = lab2.rename(columns={'Prelab-result': 'Prelab-result2', 'Inlab-result': 'Inlab-result2'})
lab3 = lab3.rename(columns={'Prelab-result': 'Prelab-result3', 'Inlab-result': 'Inlab-result3'})

merged_df = pd.merge(lab1, lab2, on='studentID', how='inner')

merged_df = pd.merge(merged_df, lab3, on='studentID', how='inner')

merged_df = pd.merge(merged_df, df2, on='studentID', how='inner')

merged_df['Prelab-result4'] = merged_df[['Prelab-result1', 'Prelab-result2', 'Prelab-result3']].mean(axis=1)
merged_df['Inlab-result4'] = merged_df[['Inlab-result1', 'Inlab-result2', 'Inlab-result3']].mean(axis=1)

print(merged_df)
# Xuất dữ liệu ra file CSV mới file này là đầu đủ sao khi fake :V 
