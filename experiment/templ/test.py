import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设你已经通过模型计算出了三组分数
# scores_pos = model.predict(test_positive_edges)
# scores_neg = model.predict(random_negative_edges)
# scores_unlearn = model.predict(unlearning_target_edges)

# 这里模拟一些数据方便你理解
np.random.seed(42)
scores_pos = np.random.normal(loc=8.0, scale=1.5, size=1000)   # 正样本分数高
scores_neg = np.random.normal(loc=-2.0, scale=1.5, size=1000)  # 负样本分数低

# 情况A：遗忘前 (Before Unlearning) - 遗忘目标看起来像正样本
scores_unlearn_before = np.random.normal(loc=7.5, scale=1.5, size=200)

# 情况B：遗忘后 (After Unlearning) - 遗忘目标应该看起来像负样本
scores_unlearn_after = np.random.normal(loc=-1.8, scale=1.5, size=200)

print(scores_pos.shape, scores_neg.shape, scores_unlearn_before.shape, scores_unlearn_after.shape)
print(type(scores_pos), type(scores_neg), type(scores_unlearn_before), type(scores_unlearn_after))
print(scores_pos[:5], scores_neg[:5], scores_unlearn_before[:5], scores_unlearn_after[:5])
def plot_distribution(pos, neg, unlearn, title):
    plt.figure(figsize=(8, 5))
    
    # 绘制 Positive Edges (蓝色)
    sns.kdeplot(pos, fill=True, color='blue', label='Positive Edges', alpha=0.3)
    
    # 绘制 Negative Edges (绿色)
    sns.kdeplot(neg, fill=True, color='green', label='Negative Edges', alpha=0.3)
    
    # 绘制 Unlearning Edges (橙色)
    sns.kdeplot(unlearn, fill=True, color='orange', label='Unlearned/Adversarial Edges', alpha=0.5)
    
    # 添加均值虚线 (参考论文图示)
    plt.axvline(np.mean(pos), color='blue', linestyle='--', alpha=0.6)
    plt.axvline(np.mean(neg), color='green', linestyle='--', alpha=0.6)
    plt.axvline(np.mean(unlearn), color='orange', linestyle='--', alpha=0.6)

    plt.title(title)
    plt.xlabel('Link Prediction Score')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 模拟画图
plot_distribution(scores_pos, scores_neg, scores_unlearn_before, "Distribution Before Unlearning")
plot_distribution(scores_pos, scores_neg, scores_unlearn_after, "Distribution After Unlearning (Ours)")