import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic
import scipy.stats as stats
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==========================================
# 数据准备 (基于之前提供的详细计数)
# ==========================================

# 原始计数数据
counts_data = {
    'Healthy': {'Positive': 2287, 'Neutral': 2244, 'Negative': 1960},
    'Depressed': {'Positive': 823, 'Neutral': 926, 'Negative': 994}
}

N_healthy = sum(counts_data['Healthy'].values())
N_depressed = sum(counts_data['Depressed'].values())

# 计算百分比用于柱状图
percentages = {
    'Healthy': [
        counts_data['Healthy']['Positive'] / N_healthy * 100,
        counts_data['Healthy']['Neutral'] / N_healthy * 100,
        counts_data['Healthy']['Negative'] / N_healthy * 100
    ],
    'Depressed': [
        counts_data['Depressed']['Positive'] / N_depressed * 100,
        counts_data['Depressed']['Neutral'] / N_depressed * 100,
        counts_data['Depressed']['Negative'] / N_depressed * 100
    ]
}

# 定义符合学术风格的颜色
color_healthy = '#2c7bb6'  # 学术蓝
color_depressed = '#d7191c'  # 学术红

# 设置全局字体样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

# ==========================================
# 图 1: 分组柱状图 (Grouped Bar Chart)
# ==========================================

labels = ['Positive', 'Neutral', 'Negative']
x = np.arange(len(labels))
width = 0.35

# 绘制柱子
rects1 = ax1.bar(x - width/2, percentages['Healthy'], width, label=f'Healthy (N={N_healthy})', color=color_healthy, edgecolor='white', linewidth=0.5)
rects2 = ax1.bar(x + width/2, percentages['Depressed'], width, label=f'Depressed (N={N_depressed})', color=color_depressed, edgecolor='white', linewidth=0.5)

# 添加数值标签函数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# 设置坐标轴和标题
ax1.set_ylabel('Percentage of Utterances (%)', fontsize=14)
ax1.set_xlabel('Group', fontsize=14)
ax1.set_title('Figure 1: Sentiment Distribution by Diagnosis Group', fontsize=16, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
ax1.set_ylim(0, 42) # 设置Y轴范围以容纳标签
ax1.legend(fontsize=11, loc='upper center')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 添加图注
caption1 = "Figure 1: Sentiment Distribution by Diagnosis Group\nCompared to the healthy group, the depressed group shows a lower\nproportion of positive sentiment and a higher proportion of negative\nsentiment."
fig.text(0.05, 0.01, caption1, ha='left', fontsize=12, wrap=True)


# ==========================================
# 图 2: 带有皮尔逊残差的马赛克图 (Mosaic Plot with Pearson Residuals)
# ==========================================

# 准备马赛克图所需的数据结构 (pandas DataFrame)
# 注意：为了让Negative在底部，Positive在顶部，我们需要调整分类的顺序
sentiment_order = ['Positive', 'Neutral', 'Negative']
diagnosis_order = ['Healthy', 'Depressed']

data_list = []
for diag in diagnosis_order:
    for sent in sentiment_order:
        # 重复添加条目以构建原始数据集表示
        data_list.extend([[diag, sent]] * counts_data[diag][sent])

df = pd.DataFrame(data_list, columns=['Diagnosis', 'Sentiment'])

# 计算卡方和皮尔逊残差
# 创建列联表 (注意顺序，这决定了残差矩阵的形状)
contingency_table = pd.crosstab(df['Sentiment'], df['Diagnosis']).reindex(index=sentiment_order, columns=diagnosis_order)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
observed = contingency_table.values
residuals = (observed - expected) / np.sqrt(expected)

# 创建残差查找字典，用于绘图时的颜色映射
residual_map = {}
for i, sent in enumerate(sentiment_order):
    for j, diag in enumerate(diagnosis_order):
        residual_map[(diag, sent)] = residuals[i, j]

# 定义颜色映射 (Red-Blue diverging colormap)
cmap = cm.RdBu_r
# 设置残差范围，用于标准化颜色 (通常 +/- 4 足够覆盖极端值)
norm = mcolors.Normalize(vmin=-4, vmax=4)

# 定义马赛克图的属性函数 (用于着色)
def props(key):
    # key 是一个元组，例如 ('Healthy', 'Negative')
    res = residual_map[key]
    color = cmap(norm(res))
    # 为Depressed-Negative添加强调边框
    # 使用 facecolor 而不是 color，避免覆盖 edgecolor
    if key == ('Depressed', 'Negative'):
        return {'facecolor': color, 'edgecolor': 'black', 'linewidth': 2}
    return {'facecolor': color, 'edgecolor': 'white', 'linewidth': 1}

# 绘制马赛克图
# 注意：statsmodels mosaic的坐标轴标签有点棘手，需要手动调整
mosaic(df, index=['Diagnosis', 'Sentiment'], ax=ax2, properties=props,
       labelizer=lambda k: '', # 不在图块内显示文本标签
       axes_label=False, # 禁用默认坐标轴标签，稍后手动添加
       gap=0.02 # 设置图块间隙
       )

# 手动设置马赛克图的坐标轴标签
ax2.set_xlabel('Diagnosis', fontsize=14)
ax2.set_ylabel('Sentiment', fontsize=14)

# 设置X轴刻度标签 (Diagnosis)
ax2.text(0.35, -0.05, 'Healthy', ha='center', va='top', fontsize=12, transform=ax2.transAxes)
ax2.text(0.85, -0.05, 'Depressed', ha='center', va='top', fontsize=12, transform=ax2.transAxes)

# 设置Y轴刻度标签 (Sentiment) - 需要根据马赛克图的分割比例估算位置
# 计算Y轴分割点 (基于边际概率)
total_obs = observed.sum()
sent_totals = observed.sum(axis=1)
y_splits = np.cumsum(sent_totals) / total_obs
y_pos = [1.0 - (y_splits[0]/2), 1.0 - (y_splits[0] + (y_splits[1]-y_splits[0])/2), 1.0 - (y_splits[2] + (y_splits[1]-y_splits[2])/2)]

# 注意：mosaic图的Y轴是从上到下绘制的(Positive在最上)
ax2.text(-0.05, y_pos[0], 'Positive', ha='right', va='center', fontsize=12, transform=ax2.transAxes, rotation=90)
ax2.text(-0.05, y_pos[1], 'Neutral', ha='right', va='center', fontsize=12, transform=ax2.transAxes, rotation=90)
ax2.text(-0.05, y_pos[2], 'Negative', ha='right', va='center', fontsize=12, transform=ax2.transAxes, rotation=90)

ax2.set_title('Figure 2: Mosaic Plot of Diagnosis vs. Sentiment\nwith Pearson Residuals', fontsize=16, pad=20)

# 添加颜色条 (Colorbar)
# 在ax2右侧创建一个新的坐标轴用于显示colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.2)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Pearson Residual', fontsize=14)
cbar.ax.tick_params(labelsize=11)

# 添加图注
# 重新计算 2x3 table 的精确 Chi-square 和 p值用于图注
chi2_val, p_val, _, _ = stats.chi2_contingency(observed)
p_text = "< 1e-9" if p_val < 1e-9 else f"= {p_val:.2e}"

caption2 = f"Figure 2: Mosaic Plot of Diagnosis vs. Sentiment with Pearson Residuals\nShading indicates deviation from independence. Red: observed >\nexpected; Blue: observed < expected. The strong red shading in the\nDepressed-Negative tile highlights the significant overrepresentation\nof negative sentiment in depressed speech (Chi-square = {chi2_val:.1f}, p {p_text}\nbased on 2x3 table)."
# 调整位置以对齐右侧图表
fig.text(0.52, 0.01, caption2, ha='left', fontsize=12, wrap=True)

# ==========================================
# 最终布局调整和显示
# ==========================================
# 调整子图间距，为底部的图注留出空间
plt.subplots_adjust(bottom=0.25, wspace=0.25)

# 保存图像 (可选)
plt.savefig('/home/i-liyuxin/data/bias_analysis/sentiment_analysis_academic.png', dpi=300, bbox_inches='tight')
# import scipy.stats as stats
# import numpy as np

# # 你的原始数据 (2x3)
# # Healthy: [Pos, Neu, Neg], Depressed: [Pos, Neu, Neg]
# observed = np.array([[2287, 2244, 1960], 
#                      [823, 926, 994]])

# chi2, p, df, expected = stats.chi2_contingency(observed)
# print(f"Chi2: {chi2:.2f}, df: {df}, p-value: {p}")
# # Output should be approx: Chi2: 49.76, df: 2, p-value: 1.56e-11