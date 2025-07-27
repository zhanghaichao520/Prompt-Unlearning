import numpy as np
import matplotlib.pyplot as plt

# 设置LaTeX字体
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


fig = plt.figure(figsize=(7.5,6))

# 数据
x = np.arange(0.1, 2.0, 0.2)
y1 = np.array([0.7421, 0.7444, 0.7448, 0.7448, 0.7449, 0.7453, 0.7456, 0.7458, 0.7453, 0.7458])
y3 = np.array([0.8396, 0.8376, 0.8435, 0.8439, 0.8422, 0.8415, 0.8395, 0.8376, 0.8361, 0.8357])

std1 = np.array([0.001, 0.0013, 0.0008, 0.0003, 0.001, 0.0004, 0.0007, 0.0007, 0.001, 0.0006])
std3 = np.array([0.0007, 0.001, 0.0014, 0.0009, 0.0009, 0.0004, 0.0017, 0.0026, 0.0011, 0.001])

# 绘制第一个子图
plt.subplot(221)
plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
plt.plot(x, y1, marker='^', label='AUC', zorder=10, color='royalblue')
plt.fill_between(x, y1 - std1, y1 + std1, alpha=0.2, zorder=10, color='lightsteelblue')
plt.title('MovieLens')
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel('AUC')
#plt.legend()

# 绘制第二个子图
plt.subplot(222)
plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
plt.plot(x, y3, marker='^', label='AUC', zorder=10, color='royalblue')
plt.fill_between(x, y3 - std3, y3 + std3, alpha=0.2, zorder=10, color='lightsteelblue')
plt.title('Electronics')
plt.xlabel(r'$\tau$', fontsize=12)
plt.ylabel('AUC')
#plt.legend()

##############################
# 数据
x = np.arange(5, 31, 5)
y1 = np.array([0.7409, 0.7349, 0.7431, 0.7458, 0.7429, 0.7412])
y3 = np.array([0.8396, 0.8405, 0.8425, 0.8444, 0.8423, 0.8443])

# 计算标准差和置信区间
std1 = np.array([0.0017, 0.0035, 0.0013, 0.0006, 0.0015, 0.0031])
std3 = np.array([0.0004, 0.0009, 0.0013, 0.0009, 0.0010, 0.0010])

# 绘制第三个子图
plt.subplot(223)
plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
plt.plot(x, y1, marker='^', label='AUC', zorder=10, color='crimson')
plt.fill_between(x, y1 - std1, y1 + std1, alpha=0.2, zorder=10, color='thistle')
plt.xlabel(r'$\widetilde{n}$', fontsize=12)
plt.xticks(x)
plt.ylabel('AUC')
#plt.legend()

# 绘制第四个子图
plt.subplot(224)
plt.grid(linestyle = '--', linewidth = 0.5, zorder=0)
plt.plot(x, y3, marker='^', label='AUC', zorder=10, color='crimson')
plt.fill_between(x, y3 - std3, y3 + std3, alpha=0.2, zorder=10, color='thistle')
plt.xlabel(r'$\widetilde{n}$', fontsize=12)
plt.ylabel('AUC')
#plt.legend()
plt.xticks(x)

fig.subplots_adjust(wspace=0.3, hspace=0.25)

#plt.savefig('param_tau_n_final.pdf', format='pdf', dpi=800, bbox_inches='tight')

plt.show()
