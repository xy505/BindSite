import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
#
# # 确保已安装 simhei.ttf 或其他支持中文字符的字体
# zh_font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
#
# # 定义动态掩码迭代轮次数据
# x = np.array([3, 5, 7, 9, 11])  # 要显示的x轴刻度值
# y = np.array([0.6430, 0.6535, 0.7145, 0.6639, 0.6391])
#
# # 颜色映射（基于F1值，保持与其他图表一致）
# normalize = plt.Normalize(y.min(), y.max())
# cmap = plt.get_cmap('jet')
# colors = cmap(normalize(y))
#
# # 核心逻辑：先画节点（上层），后画连线（下层），避免线穿节点
# sc = plt.scatter(x, y, c=y, cmap=cmap, s=80, edgecolors='black', linewidth=0.5, zorder=5)  # 节点带黑边，在上层
# plt.plot(x, y, color='gray', linewidth=1.5, zorder=1)  # 灰色连线，在下层不抢戏
#
# # ========== 关键修改：强制设置x轴刻度为x数组的数值 ==========
# plt.xticks(
#     ticks=x,  # 刻度位置：对应迭代轮次的x坐标
#     labels=[str(i) for i in x],  # 刻度标签：显示3、5、7、9、11（转字符串避免格式问题）
#     fontproperties=zh_font,  # 适配中文字体（即使标签是数字，统一字体更美观）
#     fontsize=10  # 刻度字体大小
# )
#
# # 添加颜色条（标注F1值，保持格式统一）
# cbar = plt.colorbar(sc)
# # cbar.set_label('F1', fontproperties=zh_font)
#
# # 标签与格式设置（适配迭代轮次数据特点）
# plt.xlabel('T', fontproperties=zh_font, fontsize=11)
# plt.ylabel('F1', fontproperties=zh_font, fontsize=11)
# plt.xlim(2, 12)  # 沿用原x轴范围，贴合轮次取值（3~11）
# plt.ylim(0.5, 0.8)  # 缩小y轴范围，突出F1值波动
# plt.grid(True, alpha=0.3)  # 半透明网格，不干扰主体视觉
#
# # 可选：优化布局，避免标签重叠
# plt.tight_layout()
#
# plt.show()
#
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
#
# # 确保已安装 simhei.ttf 或其他支持中文字符的字体
# zh_font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
#
# # 定义图卷积层数数据
# x = np.array([1, 3, 5, 7, 9])
# y = np.array([0.6484, 0.7145, 0.7006, 0.6550, 0.6480])
#
# # 颜色映射（基于F1值）
# normalize = plt.Normalize(y.min(), y.max())
# cmap = plt.get_cmap('jet')
# colors = cmap(normalize(y))
#
# # 关键逻辑：先画节点（上层），后画连线（下层），避免线穿节点
# sc = plt.scatter(x, y, c=y, cmap=cmap, s=80, edgecolors='black', linewidth=0.5, zorder=5)  # 节点带黑边，在上层
# plt.plot(x, y, color='gray', linewidth=1.5, zorder=1)  # 灰色连线，在下层
#
# # ========== 核心修改：强制设置x轴刻度为图卷积层数的具体值 ==========
# plt.xticks(
#     ticks=x,  # 刻度位置：精准对齐1、3、5、7、9
#     labels=[str(i) for i in x],  # 刻度显示的文字
#     fontproperties=zh_font,  # 统一使用中文字体，避免数字字体混搭
#     fontsize=10,  # 刻度字体大小（可根据需要调整）
#     fontweight='bold'  # 可选：加粗刻度，更醒目
# )
#
# # 添加颜色条（标注F1值）
# cbar = plt.colorbar(sc)
# # cbar.set_label('F1', fontproperties=zh_font, fontsize=10)  # 可选：给颜色条加标签
#
# # 标签与格式设置（适配图卷积层数的取值特点）
# plt.xlabel('l', fontproperties=zh_font, fontsize=11, fontweight='bold')
# plt.ylabel('F1', fontproperties=zh_font, fontsize=11, fontweight='bold')
# plt.xlim(0, 10)  # 适配层数范围（1~9），左右留空更美观
# plt.ylim(0.5, 0.8)  # 缩小y轴范围，突出F1值波动
# plt.grid(True, alpha=0.3)  # 半透明网格，不干扰主体视觉
#
# # 可选：优化布局，避免标签重叠或被裁剪
# plt.tight_layout()
#
# plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
#
# # 确保已安装 simhei.ttf 或其他支持中文字符的字体
# zh_font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
#
# # 定义动态掩码权重数据
# x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
# y = np.array([0.6115, 0.6405, 0.6897, 0.7145, 0.6340])
#
# # 颜色映射（基于F1值）
# normalize = plt.Normalize(y.min(), y.max())
# cmap = plt.get_cmap('jet')
# colors = cmap(normalize(y))
#
# # 关键逻辑：先画节点（上层），后画连线（下层），避免线穿节点
# sc = plt.scatter(x, y, c=y, cmap=cmap, s=80, edgecolors='black', linewidth=0.5, zorder=5)  # 节点在上，带黑边
# plt.plot(x, y, color='gray', linewidth=1.5, zorder=1)  # 连线在下，灰色不抢戏
#
# # ========== 核心修改：强制设置x轴刻度为动态掩码权重的具体值 ==========
# plt.xticks(
#     ticks=x,  # 刻度位置：精准对齐0.1、0.3、0.5、0.7、0.9
#     labels=[f'{i:.1f}' for i in x],  # 刻度标签保留1位小数，显示更规范
#     fontproperties=zh_font,  # 统一使用中文字体，避免数字字体混搭
#     fontsize=10,  # 刻度字体大小（可按需调整）
#     fontweight='bold'  # 可选：加粗刻度，更醒目
# )
#
# # 添加颜色条（标注F1值）
# cbar = plt.colorbar(sc)
# # cbar.set_label('F1', fontproperties=zh_font, fontsize=10)  # 可选：给颜色条加标签
#
# # 标签与格式设置（匹配原数据的x轴含义）
# plt.xlabel('α', fontproperties=zh_font, fontsize=11, fontweight='bold')
# plt.ylabel('F1', fontproperties=zh_font, fontsize=11, fontweight='bold')
# plt.xlim(0, 1)  # 适配动态掩码权重的取值范围（0~1）
# plt.ylim(0.5, 0.8)  # 缩小y轴范围，突出F1值差异
# plt.grid(True, alpha=0.3)  # 网格半透明，不干扰视觉
#
# # 可选：优化布局，避免标签重叠或被裁剪
# plt.tight_layout()
#
# plt.show()
#
# # 多头注意力头数
# # 确保已安装 simhei.ttf 或其他支持中文字符的字体
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
#
# # 确保已安装 simhei.ttf 或其他支持中文字符的字体
# zh_font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')
#
# # 定义数据
# x = np.array([2, 4, 8, 12, 16])
# y = np.array([0.6712, 0.6965, 0.7126, 0.7041, 0.6948])
#
# # 颜色映射
# normalize = plt.Normalize(y.min(), y.max())
# cmap = plt.get_cmap('jet')
# colors = cmap(normalize(y))
#
# # 关键修改：先画节点，后画连线（节点覆盖连线）
# sc = plt.scatter(x, y, c=y, cmap=cmap, s=80, edgecolors='black', linewidth=0.5, zorder=5)  # zorder=5 确保节点在最上层
# plt.plot(x, y, color='gray', linewidth=1.5, zorder=1)  # zorder=1 让连线在下层
#
# # 添加颜色条
# cbar = plt.colorbar(sc)
# # cbar.set_label('F1', fontproperties=zh_font)
#
# # 标签与格式
# plt.xlabel('h', fontproperties=zh_font, fontsize=11)
# plt.ylabel('F1', fontproperties=zh_font, fontsize=11)
# plt.xlim(0, 18)
# plt.ylim(0.5, 0.8)
# plt.grid(True, alpha=0.3)
#
# plt.show()


# # 参数对比
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 确保已安装 simhei.ttf 或其他支持中文字符的字体
zh_font = font_manager.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')

# ===================== 第一步：定义所有数据并计算全局颜色范围 =====================
# 1. 动态掩码迭代轮次数据
x1 = np.array([3, 5, 7, 9, 11])
y1 = np.array([0.6430, 0.6535, 0.7145, 0.6639, 0.6391])

# 2. 图卷积层数数据
x2 = np.array([1, 3, 5, 7, 9])
y2 = np.array([0.6484, 0.7145, 0.7006, 0.6550, 0.6480])

# 3. 动态掩码权重数据
x3 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
y3 = np.array([0.6115, 0.6405, 0.6897, 0.7145, 0.6340])

# 4. 多头注意力头数数据
x4 = np.array([2, 4, 8, 12, 16])
y4 = np.array([0.6712, 0.6965, 0.7126, 0.7041, 0.6948])

# 计算所有y值的全局最大/最小值（统一颜色条范围）
all_y = np.concatenate([y1, y2, y3, y4])
global_min = all_y.min()
global_max = all_y.max()

# 统一的颜色映射（所有图共用）
normalize = plt.Normalize(global_min, global_max)
cmap = plt.get_cmap('jet')

# ===================== 第二步：绘制第一张图（动态掩码迭代轮次） =====================
plt.figure(figsize=(6, 4))
sc1 = plt.scatter(x1, y1, c=y1, cmap=cmap, norm=normalize, s=80, edgecolors='black', linewidth=0.5, zorder=5)
plt.plot(x1, y1, color='gray', linewidth=1.5, zorder=1)

# x轴设置（保留原有逻辑）
plt.xticks(ticks=x1, labels=[str(i) for i in x1], fontproperties=zh_font, fontsize=10)
plt.xlabel('T', fontproperties=zh_font, fontsize=11)
plt.ylabel('F1', fontproperties=zh_font, fontsize=11)
plt.xlim(2, 12)
plt.ylim(0.5, 0.8)
plt.grid(True, alpha=0.3)

cbar1 = plt.colorbar(sc1)
plt.tight_layout()
plt.show()

# ===================== 第三步：绘制第二张图（图卷积层数） =====================
plt.figure(figsize=(6, 4))
sc2 = plt.scatter(x2, y2, c=y2, cmap=cmap, norm=normalize, s=80, edgecolors='black', linewidth=0.5, zorder=5)
plt.plot(x2, y2, color='gray', linewidth=1.5, zorder=1)

# x轴设置（保留原有逻辑）
plt.xticks(ticks=x2, labels=[str(i) for i in x2], fontproperties=zh_font, fontsize=10, fontweight='bold')
plt.xlabel('l', fontproperties=zh_font, fontsize=11, fontweight='bold')
plt.ylabel('F1', fontproperties=zh_font, fontsize=11, fontweight='bold')
plt.xlim(0, 10)
plt.ylim(0.5, 0.8)
plt.grid(True, alpha=0.3)

cbar2 = plt.colorbar(sc2)
plt.tight_layout()
plt.show()

# ===================== 第四步：绘制第三张图（动态掩码权重） =====================
plt.figure(figsize=(6, 4))
sc3 = plt.scatter(x3, y3, c=y3, cmap=cmap, norm=normalize, s=80, edgecolors='black', linewidth=0.5, zorder=5)
plt.plot(x3, y3, color='gray', linewidth=1.5, zorder=1)

# x轴设置（保留原有逻辑）
plt.xticks(ticks=x3, labels=[f'{i:.1f}' for i in x3], fontproperties=zh_font, fontsize=10, fontweight='bold')
plt.xlabel('α', fontproperties=zh_font, fontsize=11, fontweight='bold')
plt.ylabel('F1', fontproperties=zh_font, fontsize=11, fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0.5, 0.8)
plt.grid(True, alpha=0.3)

cbar3 = plt.colorbar(sc3)
plt.tight_layout()
plt.show()

# ===================== 第五步：绘制第四张图（多头注意力头数） =====================
plt.figure(figsize=(6, 4))
sc4 = plt.scatter(x4, y4, c=y4, cmap=cmap, norm=normalize, s=80, edgecolors='black', linewidth=0.5, zorder=5)
plt.plot(x4, y4, color='gray', linewidth=1.5, zorder=1)

# ========== 关键修改：x轴刻度设为2到18，间隔2 ==========
plt.xticks(
    ticks=np.arange(2, 17, 2),  # 生成2,4,...,18的刻度
    labels=[str(i) for i in np.arange(2, 17, 2)],
    fontproperties=zh_font,
    fontsize=10
)
plt.xlabel('h', fontproperties=zh_font, fontsize=11)
plt.ylabel('F1', fontproperties=zh_font, fontsize=11)
plt.xlim(0, 18)
plt.ylim(0.5, 0.8)
plt.grid(True, alpha=0.3)

cbar4 = plt.colorbar(sc4)
plt.tight_layout()
plt.show()








# # 消融实验柱状图
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
# import matplotlib as mpl
#
# # ---------------------- 全局高清渲染配置（核心优化）----------------------
# # 设置全局DPI（匹配导出DPI，避免缩放导致模糊）
# mpl.rcParams['figure.dpi'] = 300
# mpl.rcParams['savefig.dpi'] = 300
# # 抗锯齿渲染（解决文字模糊/锯齿）
# mpl.rcParams['agg.path.chunksize'] = 10000
# mpl.rcParams['text.antialiased'] = True
# mpl.rcParams['font.family'] = 'sans-serif'
#
# # 配置中文字体（优先系统级字体，确保渲染清晰）
# # 方式1：指定绝对路径（Windows）
# zh_font_path = 'C:/Windows/Fonts/simhei.ttf'
# # 方式2：Linux/Mac（注释掉上面，启用下面）
# # zh_font_path = '/System/Library/Fonts/PingFang.ttc'
# zh_font = font_manager.FontProperties(fname=zh_font_path, size=12)
# # 全局字体大小适配
# mpl.rcParams['font.size'] = 11  # 基础字体大小提升
# mpl.rcParams['axes.labelsize'] = 14
# mpl.rcParams['xtick.labelsize'] = 12
# mpl.rcParams['ytick.labelsize'] = 12
# mpl.rcParams['legend.fontsize'] = 12
# mpl.rcParams['axes.titlesize'] = 16
#
# # 解决负号显示问题
# mpl.rcParams['axes.unicode_minus'] = False
#
# # ---------------------- 消融实验数据 ----------------------
# metrics = ['PRE', 'REC', 'SPE', 'F1', 'ACC', 'MCC', 'AUC', 'AUPRC']  # x轴标签
# groups = ['DMGNN', 'w/o PL', 'w/o WA', 'w/o DM', 'w/o ML']  # 不同颜色对应的模型
# data = np.array([
#     [0.6573, 0.8034, 0.9479, 0.7268, 0.9231, 0.6513, 0.9442, 0.7701],  # 完整模型
#     [0.5671, 0.8214, 0.9030, 0.6752, 0.9012, 0.5794, 0.9233, 0.7544],    # w/o PL
#     [0.5713, 0.8011, 0.9077, 0.6845, 0.9002, 0.6337, 0.9433, 0.7489],    # w/o WA
#     [0.5949, 0.7938, 0.9229, 0.6973, 0.9057, 0.6233, 0.9410, 0.7589],    # w/o DM
#     [0.5849, 0.8132, 0.9112, 0.7197, 0.9188, 0.6237, 0.9316, 0.7566]     # w/o ML
# ])
#
# # ---------------------- 图表样式配置 ----------------------
# bar_width = 0.15  # 保持原有柱宽
# # colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']  # 专业配色
# colors = ['#525254', '#6CAD41', '#F4811F', '#34A5CE', '#C0BFC3']
# x = np.arange(len(metrics))
# # 增大画布尺寸（适配高清文字，避免挤压）
# fig, ax = plt.subplots(figsize=(16, 9))
#
# # 绘制柱状图+添加数值标签（优化标签字体）
# for i, (group, color) in enumerate(zip(groups, colors)):
#     bars = ax.bar(x + i * bar_width, data[i, :], width=bar_width, label=group,
#                   color=color, alpha=0.8, edgecolor='black', linewidth=0.8)  # 边框略加粗，提升轮廓
#
#     # 可选：添加数值标签（优化字体大小和权重）
#     # for bar, value in zip(bars, data[i, :]):
#     #     height = bar.get_height()
#     #     ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
#     #             f'{value:.4f}',
#     #             ha='center', va='bottom', fontsize=9, fontweight='bold',
#     #             fontproperties=zh_font)  # 强制使用清晰中文字体
#
# # ---------------------- 图表标注与格式优化 ----------------------
# # X轴配置
# ax.set_xlabel('评价指标', fontproperties=zh_font, fontweight='bold', labelpad=10)
# ax.set_xticks(x + bar_width * 2)
# ax.set_xticklabels(metrics, fontproperties=zh_font)  # 强制使用清晰字体
#
# # Y轴配置
# ax.set_ylabel('评价指标值', fontproperties=zh_font, fontweight='bold', labelpad=10)
# ax.set_ylim(0.4, 1.05)
# ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
# ax.spines['top'].set_visible(False)  # 隐藏顶部边框，视觉更简洁
# ax.spines['right'].set_visible(False)  # 隐藏右侧边框
#
# # 图例优化（更大的点击区域，清晰字体）
# ax.legend(
#     loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,
#     frameon=True, fancybox=True, shadow=True,
#     prop=zh_font,  # 强制使用清晰中文字体
#     framealpha=1.0,  # 图例背景不透明，提升文字对比度
#     borderpad=1.0,  # 图例内边距，避免文字挤压
#     labelspacing=0.8
# )
#
# # 标题优化（更大字体，增加间距）
# ax.set_title(
#     '蛋白质结合位点预测模型消融实验结果',
#     fontproperties=zh_font, fontweight='bold',
#     pad=25
# )
#
# # ---------------------- 布局与导出优化 ----------------------
# # 精细化布局调整（避免文字裁剪）
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.18, top=0.92, left=0.08, right=0.98)
#
# # 导出配置（核心：无损高清）
# plt.savefig(
#     '消融实验结果柱状图_高清版.png',
#     dpi=300,
#     bbox_inches='tight',  # 自动裁剪空白，但保留完整内容
#     facecolor='white',
#     edgecolor='none',
#     # 关键：禁用压缩，确保文字清晰
#     pil_kwargs={'quality': 100, 'optimize': False}
# )
#
# # 显示图片（注意：Jupyter中显示可能仍略模糊，但导出文件是高清的）
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 适配学术风格：去掉边框、调整网格线
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 消融实验数据
data = np.array([
    [0.6573, 0.8534, 0.9479, 0.7268, 0.9231, 0.6513, 0.9442, 0.7701],  # DMGNN
    [0.5671, 0.8214, 0.9030, 0.6752, 0.9012, 0.5794, 0.9233, 0.7544],  # w/o PL
    [0.5713, 0.8011, 0.9077, 0.6845, 0.9002, 0.6337, 0.9433, 0.7489],  # w/o WA
    [0.5949, 0.7938, 0.9229, 0.6973, 0.9057, 0.6233, 0.9410, 0.7589],  # w/o DM
    [0.5849, 0.8132, 0.9112, 0.7197, 0.9188, 0.6237, 0.9316, 0.7566]  # w/o ML
])

# 定义标签
models = ['DMGNN', 'w/o PL', 'w/o WA', 'w/o DM', 'w/o ML']
metrics = ['PRE', 'REC', 'SPE', 'F1', 'ACC', 'MCC', 'AUC', 'AUPRC']
n_models = len(models)  # 模型数量（5个）
n_metrics = len(metrics)  # 指标数量（8个）

# 核心调整：增大组内间隔，优化柱子宽度
group_width = 0.9  # 每组占用的宽度比例（增大以利用空间）
bar_width = 0.15  # 固定单柱子宽度（不再自动计算，保证组内间隔）
gap = 0.02  # 组内柱子之间的额外间隔

# 计算每个指标组的基准位置
x = np.arange(n_metrics)

# 顶刊论文主流配色（NeurIPS/ICML/CVPR）
# colors = [
#     '#1f77b4',  # 主蓝色（核心模型，视觉焦点）
#     '#ff7f0e',  # 橙色
#     '#2ca02c',  # 绿色
#     '#d62728',  # 红色
#     '#9467bd'   # 紫色
# ]

colors = ['#FAAF7F', '#BAB4D5', '#F993BD', '#D7B98E','#FCD17D']

# 绘制图表（增加画布高度，适配更大的字体）
fig, ax = plt.subplots(figsize=(18, 10))

# 绘制每个模型的柱子并添加数值标签
for i in range(n_models):
    # 计算每个柱子的位置：增加组内间隔
    bar_positions = x - (group_width / 2) + (i * (bar_width + gap)) + (gap / 2)

    # 绘制柱子（顶刊风格）
    bars = ax.bar(bar_positions, data[i], bar_width,
                  label=models[i], color=colors[i], alpha=0.85,
                  edgecolor='white', linewidth=1.2)

    # 柱子顶部垂直数值标签（同步放大字体）
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.008,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=15, fontweight='medium',  # 数值字体放大
                rotation=90)

# 风格优化 - 核心字体放大调整
# 标题字体放大（顶刊标题通常更大）
ax.set_title('',
             fontsize=18, fontweight='bold', pad=25)

# x轴：指标名称字体大幅放大（核心修改）
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=16,)  # 指标名放大到16号加粗
ax.tick_params(axis='x', length=0, labelsize=14)

# y轴刻度字体放大
ax.set_ylim(0.5, 1.05)
ax.set_yticks(np.arange(0.5, 1.01, 0.1))
ax.tick_params(axis='y', labelsize=16)  # y轴刻度放大到14号
ax.tick_params(axis='x', labelsize=16)  # y轴刻度放大到14号
ax.grid(axis='y', alpha=0.25, linestyle='-', linewidth=1)

# 图例（图片下方注释）字体大幅放大
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          frameon=False, fontsize=20,  # 图例字体放大到15号
          ncol=n_models, columnspacing=4)

# 若后续需要显示坐标轴标签，也同步放大（当前为空，预留配置）
ax.set_xlabel('', fontsize=16, fontweight='bold', labelpad=25)
ax.set_ylabel('', fontsize=16, fontweight='bold', labelpad=15)

# 添加组间分隔线
for i in range(1, n_metrics):
    ax.axvline(x=i - 0.5, color='lightgray', linestyle='-', linewidth=0.6)

# 调整布局，适配更大的字体和下方图例
plt.subplots_adjust(bottom=0.2)  # 增大底部留白，避免图例截断
# 保存高清图
plt.savefig('消融实验分组柱状图_顶刊风格_大字体.png', dpi=300, bbox_inches='tight')
plt.show()

# # 鲁棒性实验画图
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设置中文字体（避免乱码）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 适配学术风格：精简边框、优化网格线
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.right'] = False
# plt.rcParams['axes.spines.left'] = True
# plt.rcParams['axes.spines.bottom'] = True
# plt.rcParams['axes.linewidth'] = 0.8  # 保留的边框更细，更精致
#
# # 消融实验数据
# data = np.array([
#     [0.3873, 0.3184],  # GraphPPIS: Bound, Unbound
#     [0.3431, 0.3320],  # Fpocket: Bound, Unbound
#     [0.4109, 0.4019],  # P2Rank: Bound, Unbound
#     [0.4310, 0.4219],  # DMGNN: Bound, Unbound
# ])
#
# # 定义标签
# models = ['Bound', 'Unbound']
# metrics = ['GraphPPIS', 'Fpocket', 'P2Rank', 'DMGNN']
# n_models = len(models)  # 模型数量（2个）
# n_metrics = len(metrics)  # 指标数量（4个）
#
# # 核心布局调整：重点优化组内柱子间距
# group_width = 0.7  # 每组总宽度
# inner_gap = 0.1  # 组内两个柱子之间的间距（核心调整）
# bar_width = (group_width - inner_gap) / n_models  # 单柱子宽度（扣除间距后分配）
# gap_between_groups = 0.3  # 组间间距
#
# # 计算每个指标组的基准位置
# x = np.arange(n_metrics) * (group_width + gap_between_groups)
#
# # 科研感配色方案（Nature/Science期刊常用配色）
# # colors = [
# #     '#3498db',  # 亮蓝色（Bound，主色）#436BB4
# #     '#00b5e4',  # 珊瑚红（Unbound，对比色）
# # ]
# colors = [
#     '#D9A3A0',  # 亮蓝色（Bound，主色）
#     '#F1CEE4',  # 珊瑚红（Unbound，对比色）
# ]
# # 绘制图表
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 绘制每个模型的柱子并添加数值标签
# for i in range(n_models):
#     # 计算每个柱子的位置：预留组内间距
#     bar_positions = x - (group_width - inner_gap) / 2 + (i * (bar_width + inner_gap)) + bar_width / 2
#
#     # 绘制柱子
#     bars = ax.bar(
#         bar_positions, data[:, i], bar_width,
#         label=models[i], color=colors[i], alpha=0.85,
#         edgecolor='white', linewidth=1.2,
#         zorder=3
#     )
#
#     # 柱子顶部数值标签
#     for bar in bars:
#         height = bar.get_height()
#         ax.text(
#             bar.get_x() + bar.get_width() / 2., height + 0.003,
#             f'{height:.4f}',
#             ha='center', va='bottom', fontsize=12, fontweight='medium',
#             color='black',
#             fontfamily='Arial'  # 这个参数在text里是支持的
#         )
#
# # 标题和轴标签设置（科研风格）
# ax.set_title('', fontsize=14, fontweight='bold', pad=20, fontfamily='Arial')
# ax.set_xlabel('', fontsize=12, fontweight='medium', labelpad=10, fontfamily='Arial')
# ax.set_ylabel('MCC', fontsize=14, fontweight='medium', labelpad=10, fontfamily='Arial')
#
# # X轴优化
# ax.set_xticks(x)
# ax.set_xticklabels(metrics, fontsize=12, fontweight='medium', fontfamily='Arial')
# ax.tick_params(axis='x', length=0, pad=8)
#
# # Y轴优化
# y_min = 0.25
# y_max = 0.5
# ax.set_ylim(y_min, y_max)
# ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.05))
# ax.tick_params(axis='y', labelsize=10, pad=5)
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
#
# # 网格线优化
# ax.grid(axis='y', alpha=0.2, linestyle=':', linewidth=0.8, zorder=0, color='gray')
#
# # 图例优化：移除不支持的fontfamily参数（核心修复）
# # 图例优化：放大颜色标注（核心修改）
# ax.legend(
#     loc='upper right',
#     frameon=True,
#     framealpha=0.95,
#     facecolor='white',
#     edgecolor='lightgray',
#     fontsize=14,          # 放大文字
#     ncol=1,
#     handlelength=2.0,     # 放大色块长度
#     handleheight=1.0,     # 放大色块高度
#     labelspacing=0.8,     # 增大标签间距
#     borderpad=0.8,        # 增大图例内边距
#     handletextpad=0.8     # 增大色块-文字间距
# )
#
# # 添加组间分隔线
# for i in range(1, n_metrics):
#     ax.axvline(
#         x[i] - (group_width + gap_between_groups) / 2,
#         color='lightgray', linestyle='-', linewidth=0.8,
#         zorder=0
#     )
#
# # 整体布局调整
# plt.tight_layout()
#
# # 保存高清图
# plt.savefig('鲁棒性实验分组柱状图.png', dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig('鲁棒性实验分组柱状图.pdf', dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()