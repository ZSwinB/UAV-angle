import pandas as pd
import numpy as np
import os

def split_excel_file(input_file, train_ratio=0.7):
   # 读取Excel文件
   df = pd.read_excel(input_file, header=None)
   
   # 随机打乱索引
   indices = np.random.permutation(len(df))
   
   # 计算训练集大小
   train_size = int(len(df) * train_ratio)
   
   # 划分数据集
   train_indices = indices[:train_size]
   test_indices = indices[train_size:]
   
   train_data = df.iloc[train_indices]
   test_data = df.iloc[test_indices]
   
   # 确定输出文件路径
   base_dir = os.path.dirname(input_file)
   base_name = os.path.splitext(os.path.basename(input_file))[0]
   
   train_file = os.path.join(base_dir, f"{base_name}_train.xlsx")
   test_file = os.path.join(base_dir, f"{base_name}_test.xlsx")
   
   # 保存文件
   train_data.to_excel(train_file, index=False, header=False)
   test_data.to_excel(test_file, index=False, header=False)
   
   print(f"训练集大小: {len(train_data)}行，保存至: {train_file}")
   print(f"测试集大小: {len(test_data)}行，保存至: {test_file}")
   
   return train_file, test_file

if __name__ == "__main__":
   input_file = r"d:\desktop\毕设材料\6\classifier_noisy.xlsx"
   train_ratio = float(input("请输入训练集比例(0-1之间，默认0.7): ") or "0.7")
   train_file, test_file = split_excel_file(input_file, train_ratio)