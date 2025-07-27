import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 模拟数据生成
np.random.seed(0)
x_student_neg = lambda: np.random.normal(0, 2, 1000)
x_student_pos = lambda: np.random.normal(5, 2, 1000)
x_teacher_neg = lambda: np.random.normal(0, 0.5, 1000)
x_teacher_pos = lambda: np.random.normal(5, 1.5, 1000)

titles = ['ALDI', 'DeepMusic', 'GAR', 'MTPR', 'CLCRec']

fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)

for ax, title in zip(axes, titles):
    sns.kdeplot(x_student_neg(), ax=ax, fill=True, color='blue', label='Student Neg.')
    sns.kdeplot(x_student_pos(), ax=ax, fill=True, color='orange', label='Student Pos.')
    sns.kdeplot(x_teacher_neg(), ax=ax, color='green', linestyle='--', label='Teacher Neg.')
    sns.kdeplot(x_teacher_pos(), ax=ax, color='red', linestyle='--', label='Teacher Pos.')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Predicted score")
    ax.set_xlim(-10, 30 if title == 'GAR' or title == 'MTPR' else 15)
    if title == 'ALDI':
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.set_ylabel("")

plt.tight_layout()
plt.show()
