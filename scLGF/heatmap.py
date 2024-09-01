import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#加载数据集
# df = pd.read_csv("/path/to/your/folder/mtcars.csv") #注意替换成文件相应的路径
from matplotlib import ticker

# d = [[0.01,0.8185,0.8146,0.8142,0.8135,0.822],[0.1,0.82,0.8251,0.815,0.8176,0.822],[1,0.8069,0.8197,0.822,0.822,0.8176],[10,0.8087,0.8069,0.8197,0.8217,0.822],[100,0.806,0.8087,0.804,0.8197,0.822]]
d = [[0.8185,0.8146,0.8142,0.8135,0.822],[0.82,0.8251,0.815,0.8176,0.822],[0.8069,0.8197,0.822,0.822,0.8176],[0.8087,0.8069,0.8197,0.8217,0.822],[0.806,0.8087,0.804,0.8197,0.822]]

# #将第一列设置为索引
# df.set_index('Unnamed: 0', inplace=True)

# #绘制热图
# sns.heatmap(df, cmap="YlGnBu")
# plt.title('Car Data Heatmap')
# plt.show()

variables = ['0.01', '0.1', '1 ' , '10 ' , '100']
labels = ['0.01', '0.1', '1 ' , '10 ' , '100']
df = pd.DataFrame(d,columns=variables, index=labels)
fig =plt.figure(figsize=(7,6))#宽、高
ax = fig.add_subplot(1,1,1)
cax = ax.matshow(df, interpolation='nearest', cmap= 'YlGnBu')
fig.colorbar(cax)
tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_xticklabels(['']+ list(df.columns ))
ax.set_yticklabels([''] + list( df.index))
plt.show()
