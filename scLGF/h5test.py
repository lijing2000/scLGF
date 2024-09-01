from loadData import loadH5AnnData
import pandas as pd
import numpy as np
# path = "D:/研一/单细胞/scDFCN/data/Quake_10x_Limb_Muscle.h5"

# anndata = loadH5AnnData(path)
#
# print(anndata.X)



# path处填入.h5文件绝对路径
df = pd.read_hdf(r'D:/研一/单细胞/scDFCN/data/Quake_10x_Limb_Muscle.h5')
# 路径部分依旧根据自己需要设置
np.save("D:/研一/单细胞/scDFCN/data/Quake_10x_Limb_Muscle.npy", df)
pd.DataFrame(df).to_csv('D:/研一/单细胞/scDFCN/data/Quake_10x_Limb_Muscle.csv', header=None, index=None)